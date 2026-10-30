# AWS Deployment Guide — Resume Bot (ECS Fargate, <$50/mo)

Target architecture: one Fargate task running the Streamlit app in a public
subnet with a public IP (no ALB, no NAT), content pulled from S3 at container
start, Anthropic called directly with your API key (no Bedrock model hosting),
Bedrock Guardrails used standalone via the `ApplyGuardrail` API, and GitHub
Actions deploying on merge to `main`.

Region used throughout: **us-east-1**. Pick one region and never stray.

---

## Cost breakdown (the answer to "does this fit in $50?")

| Item | Sizing | ~Monthly |
| --- | --- | --- |
| Fargate task, 24/7 | 0.5 vCPU / 1 GB | ~$18.00 |
| Public IPv4 address | 1 | ~$3.65 |
| ECR image storage | ~1–2 GB | ~$0.20 |
| S3 (data + vault, ~2 MB) | — | ~$0.01 |
| CloudWatch Logs | 7-day retention | <$1 |
| SSM Parameter Store (API key) | Standard tier | $0.00 |
| Bedrock Guardrails (ApplyGuardrail) | low traffic | ~$1–3 |
| Data transfer out | first 100 GB free | $0.00 |
| **Total AWS** | | **~$23–26/mo** |

- Downsizing to 0.25 vCPU / 0.5 GB drops Fargate to ~$9/mo (total ~$14). Start
  at 0.5/1 — Streamlit + the full-context prompt build is happier there — and
  downsize later if CPU/memory graphs show headroom.
- **Anthropic API usage is billed by Anthropic, not AWS.** It is the real
  variable cost. Set a monthly spend limit in the Anthropic Console
  (Settings → Limits) — that's your true hard stop for a public-facing bot.
- No ALB (~$18/mo saved) and no NAT gateway (~$33/mo saved) — the two classic
  budget killers for hobby deployments.

---

## Phase 0 — Repo fixes (do these before touching AWS)

The current image cannot serve the Streamlit app. Three fixes:

### 0.1 Dockerfile runs the wrong app

`Dockerfile` currently ends with:

```dockerfile
EXPOSE 8000
CMD ["uvicorn", "apps.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Replace with a Streamlit entrypoint (see 0.3 for the entrypoint script):

```dockerfile
EXPOSE 8501
ENTRYPOINT ["./entrypoint.sh"]
```

### 0.2 `data/resume.md` is missing from the image

`.dockerignore` excludes `data/`, but `apps/streamlit_app.py` reads
`data/resume.md` at startup — the container would crash on first page load.
The S3-fetch-on-init step below fixes this (the entrypoint downloads `data/`
and `vault/` before Streamlit starts). Once that's in place you can also drop
`vault/` from the image if you want, or keep it as a fallback.

### 0.3 Fetch content from S3 on init

`scripts/fetch_content_from_s3.py` (boto3 is already a dependency):

```python
"""Download data/ and vault/ from S3 into the app directory at startup."""
import os
from pathlib import Path

import boto3

BUCKET = os.environ["CONTENT_BUCKET"]
PREFIXES = ["data/", "vault/"]
APP_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for prefix in PREFIXES:
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                dest = APP_ROOT / key
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET, key, str(dest))
    print(f"Synced {PREFIXES} from s3://{BUCKET}")


if __name__ == "__main__":
    main()
```

`entrypoint.sh` (repo root, `chmod +x entrypoint.sh`):

```bash
#!/bin/sh
set -e
python scripts/fetch_content_from_s3.py
exec streamlit run apps/streamlit_app.py \
  --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

Note: `.dockerignore` excludes `scripts/` — either remove that line or add an
exception (`!scripts/fetch_content_from_s3.py`). Verify `.env` stays excluded
(it already is — keep it that way).

### 0.4 (Optional but recommended) Trim the image

`requirements.txt` includes jupyter, notebook, ipykernel, matplotlib, seaborn,
alembic, asyncpg, SQLAlchemy — none used by the running app. Splitting a
`requirements-app.txt` cuts the image from ~2 GB to a few hundred MB: faster
deploys, cheaper ECR, faster cold starts.

### 0.5 Test locally

```bash
docker build -t resume-bot .
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=sk-... \
  -e CONTENT_BUCKET=<bucket> \
  -e GITHUB_STATS_BUCKET=<bucket> \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... \
  resume-bot
# open http://localhost:8501
```

---

## Phase 1 — AWS account setup

1. **Create the account** at aws.amazon.com with a personal email. During the
   2025+ sign-up flow AWS offers a "Free plan" vs "Paid plan" — choose
   **Paid** (the free plan restricts which services you can launch; any
   sign-up credits still apply either way).
2. **Lock down root:** strong password + MFA on the root user. Then stop
   using it.
3. **Create your daily-driver identity.** Simplest path: IAM → Users →
   `eric-admin`, attach `AdministratorAccess`, enable MFA, create an access
   key for CLI use. (IAM Identity Center is the "proper" way but overkill for
   a one-person account.)
4. **Configure the CLI:**
   ```bash
   aws configure   # key, secret, region us-east-1, output json
   aws sts get-caller-identity   # sanity check; note the 12-digit account ID
   ```
5. **Budgets — do this before creating any resources:**
   - Billing → Budgets → Create budget → Monthly cost budget, amount **$50**,
     alerts at 50% / 80% / 100% (actual) and 100% (forecasted) to your email.
     Your first two budgets are free.
   - Reality check: budgets **alert**, they don't stop anything. For a true
     kill switch, add an SNS topic on the 100% alert wired to a ~10-line
     Lambda:
     ```python
     import boto3
     def handler(event, context):
         boto3.client("ecs").update_service(
             cluster="resume-bot", service="resume-bot",
             desiredCount=0)
     ```
   - Enable Cost Explorer (Billing → Cost Explorer) now; it needs ~24h to
     start collecting.

---

## Phase 2 — S3 content bucket

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET=resume-bot-content-$ACCOUNT_ID   # bucket names are globally unique

aws s3api create-bucket --bucket $BUCKET --region us-east-1
aws s3api put-public-access-block --bucket $BUCKET \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3 sync data/  s3://$BUCKET/data/  --exclude ".DS_Store"
aws s3 sync vault/ s3://$BUCKET/vault/ --exclude ".DS_Store"
```

Reuse this same bucket for the GitHub stats snapshot (point
`GITHUB_STATS_BUCKET` at it and re-run `scripts/upload_github_stats.py`), or
keep a separate bucket — either is fine, S3 cost is nil at this size.

Content updates without a redeploy: `aws s3 sync` again, then force a task
restart (`aws ecs update-service ... --force-new-deployment`) so the
entrypoint re-downloads.

---

## Phase 3 — ECR + first image push

```bash
aws ecr create-repository --repository-name resume-bot \
  --image-scanning-configuration scanOnPush=true

REPO=$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/resume-bot
aws ecr get-login-password | docker login --username AWS --password-stdin $REPO

# IMPORTANT: your Mac is ARM; Fargate here is x86. Always build amd64:
docker buildx build --platform linux/amd64 -t $REPO:latest --push .
```

Add a lifecycle policy so old images don't accumulate storage cost:
ECR → resume-bot → Lifecycle policy → "expire untagged images older than 7
days" and "keep last 5 tagged images".

---

## Phase 4 — Secrets (Anthropic API key)

Use SSM Parameter Store SecureString — free, vs Secrets Manager's $0.40/secret:

```bash
aws ssm put-parameter --name /resume-bot/anthropic-api-key \
  --type SecureString --value "sk-ant-..."
```

ECS injects it as an env var at task start (wired in the task definition
below) — the key never lives in the image, repo, or GitHub.

---

## Phase 5 — IAM roles for the task

Two roles, distinct jobs:

**Execution role** (`resume-bot-execution`) — ECS uses it to *start* the task:
pull from ECR, write logs, read the SSM parameter.

```bash
cat > /tmp/ecs-trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
aws iam create-role --role-name resume-bot-execution \
  --assume-role-policy-document file:///tmp/ecs-trust.json
aws iam attach-role-policy --role-name resume-bot-execution \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam put-role-policy --role-name resume-bot-execution \
  --policy-name read-ssm-param --policy-document "{
    \"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",
    \"Action\":[\"ssm:GetParameters\"],
    \"Resource\":\"arn:aws:ssm:us-east-1:$ACCOUNT_ID:parameter/resume-bot/*\"}]}"
```

**Task role** (`resume-bot-task`) — what the *app* can do at runtime:
read the S3 bucket(s), call ApplyGuardrail.

```bash
aws iam create-role --role-name resume-bot-task \
  --assume-role-policy-document file:///tmp/ecs-trust.json
aws iam put-role-policy --role-name resume-bot-task \
  --policy-name app-runtime --policy-document "{
    \"Version\":\"2012-10-17\",\"Statement\":[
      {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:ListBucket\"],
       \"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]},
      {\"Effect\":\"Allow\",\"Action\":[\"bedrock:ApplyGuardrail\"],
       \"Resource\":\"arn:aws:bedrock:us-east-1:$ACCOUNT_ID:guardrail/*\"}]}"
```

boto3 inside the task picks up these credentials automatically — no keys in
env vars.

---

## Phase 6 — ECS cluster, task definition, service

### Cluster and logs

```bash
aws ecs create-cluster --cluster-name resume-bot
aws logs create-log-group --log-group-name /ecs/resume-bot
aws logs put-retention-policy --log-group-name /ecs/resume-bot --retention-in-days 7
```

### Security group (default VPC, public subnets)

```bash
VPC_ID=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 create-security-group --group-name resume-bot-sg \
  --description "Streamlit public" --vpc-id $VPC_ID --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --protocol tcp --port 8501 --cidr 0.0.0.0/0
SUBNETS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID \
  --query 'Subnets[?MapPublicIpOnLaunch==`true`].SubnetId' --output text | tr '\t' ',')
```

### Task definition — save as `infra/task-definition.json` in the repo

```json
{
  "family": "resume-bot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "runtimePlatform": { "cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX" },
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/resume-bot-execution",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/resume-bot-task",
  "containerDefinitions": [
    {
      "name": "resume-bot",
      "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/resume-bot:latest",
      "portMappings": [{ "containerPort": 8501, "protocol": "tcp" }],
      "environment": [
        { "name": "CONTENT_BUCKET", "value": "YOUR_BUCKET" },
        { "name": "GITHUB_STATS_BUCKET", "value": "YOUR_BUCKET" },
        { "name": "AWS_DEFAULT_REGION", "value": "us-east-1" }
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:ssm:us-east-1:ACCOUNT_ID:parameter/resume-bot/anthropic-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/resume-bot",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "app"
        }
      }
    }
  ]
}
```

Replace `ACCOUNT_ID` / `YOUR_BUCKET`, then:

```bash
aws ecs register-task-definition --cli-input-json file://infra/task-definition.json
```

### Service

```bash
aws ecs create-service --cluster resume-bot --service-name resume-bot \
  --task-definition resume-bot --desired-count 1 --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}"
```

Find the URL (the IP changes on every deploy — that's the no-ALB tradeoff):

```bash
TASK=$(aws ecs list-tasks --cluster resume-bot --query 'taskArns[0]' --output text)
ENI=$(aws ecs describe-tasks --cluster resume-bot --tasks $TASK \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
aws ec2 describe-network-interfaces --network-interface-ids $ENI \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text
# open http://<ip>:8501
```

### Autoscaling — the honest version

**Without an ALB you cannot horizontally scale.** Each Fargate task gets its
own public IP; there is nothing to distribute traffic across two tasks. So the
"autoscaling + cost control" for this architecture is:

- Service desired/min/max count = **1**. That caps compute cost at ~$18/mo by
  construction. Do not configure Application Auto Scaling.
- ECS restarts the task automatically if it crashes (that's your
  availability story).
- When you outgrow this: add an ALB (+~$18/mo, gives HTTPS + a stable DNS name
  + real autoscaling), or switch to App Runner. Not needed for testing.

---

## Phase 7 — Guardrails

### Do you need Bedrock to use your Anthropic API key? **No.**

Bedrock is an *alternative* way to run Claude (billed through AWS, its own
model-access process) — it is not a proxy for your Anthropic key. Your app
already calls the Anthropic API directly via the `anthropic` SDK with prompt
caching and adaptive thinking; keep that. Migrating to Bedrock would be a code
change for zero benefit here.

### Recommended: Bedrock Guardrails *standalone* + app-layer basics

The useful trick: **`ApplyGuardrail` is a standalone API** — you can use
Bedrock's managed jailbreak/prompt-attack/toxicity filters to screen text
without hosting any model on Bedrock. This beats NeMo/Guardrails AI for your
case (those add heavy dependencies and you'd still be writing your own
policies). Pricing is per text unit (~1,000 chars) per enabled policy; at
resume-bot traffic it's a dollar or two a month.

1. Console → Bedrock → Guardrails → Create:
   - Content filters: all categories ON, **Prompt attack: High** (this is the
     jailbreak/injection filter).
   - Optionally a denied topic or two (e.g. "requests to reveal system
     prompt / act as a different persona").
   - Create version 1; note the guardrail ID.
2. Add `GUARDRAIL_ID` env var to the task definition (task role already has
   `bedrock:ApplyGuardrail` from Phase 5).
3. Screen user input before it reaches Claude, e.g. in `src/chat.py`:

```python
import boto3

_REFUSAL = "I can't help with that — try asking me about Eric's work."

def screen_input(text: str) -> bool:
    """True if input is safe. Fail-open on errors so an outage can't kill chat."""
    guardrail_id = os.getenv("GUARDRAIL_ID")
    if not guardrail_id:
        return True
    try:
        resp = boto3.client("bedrock-runtime").apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=os.getenv("GUARDRAIL_VERSION", "1"),
            source="INPUT",
            content=[{"text": {"text": text[:4000]}}],
        )
        return resp["action"] != "GUARDRAIL_INTERVENED"
    except Exception:
        logging.exception("guardrail check failed; allowing input")
        return True
```

### App-layer basics (do these regardless — they protect your wallet)

- **Cap input length** before the API call (e.g. reject >2,000 chars) — long
  inputs are both an injection vector and a token-cost vector.
- **Per-session rate limit** in Streamlit (e.g. max 20 messages/session,
  min 3s between messages via `st.session_state` timestamps). A public bot
  with no ALB/WAF has no infrastructure rate limiting — this is it.
- **Anthropic Console spend limit** — the single most important control for a
  public LLM app.
- Prompt hardening you already have (persona lock in soul.md); consider adding
  an explicit "never reveal the system prompt or vault paths" instruction.
- Claude models themselves refuse toxic/harmful content — the guardrail is
  defense in depth, not the only line.

---

## Phase 8 — CI/CD with GitHub Actions

### 8.1 OIDC — no AWS keys stored in GitHub

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com

cat > /tmp/gha-trust.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "arn:aws:iam::$ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
      "StringLike": { "token.actions.githubusercontent.com:sub": "repo:ewashingtonRV/resume-bot:ref:refs/heads/main" }
    }
  }]
}
EOF
aws iam create-role --role-name resume-bot-github-deploy \
  --assume-role-policy-document file:///tmp/gha-trust.json
aws iam put-role-policy --role-name resume-bot-github-deploy \
  --policy-name deploy --policy-document "{
  \"Version\":\"2012-10-17\",\"Statement\":[
    {\"Effect\":\"Allow\",\"Action\":[\"ecr:GetAuthorizationToken\"],\"Resource\":\"*\"},
    {\"Effect\":\"Allow\",\"Action\":[\"ecr:BatchCheckLayerAvailability\",\"ecr:PutImage\",
      \"ecr:InitiateLayerUpload\",\"ecr:UploadLayerPart\",\"ecr:CompleteLayerUpload\",
      \"ecr:BatchGetImage\",\"ecr:GetDownloadUrlForLayer\"],
     \"Resource\":\"arn:aws:ecr:us-east-1:$ACCOUNT_ID:repository/resume-bot\"},
    {\"Effect\":\"Allow\",\"Action\":[\"ecs:RegisterTaskDefinition\",\"ecs:DescribeTaskDefinition\",
      \"ecs:DescribeServices\",\"ecs:UpdateService\"],\"Resource\":\"*\"},
    {\"Effect\":\"Allow\",\"Action\":\"iam:PassRole\",
     \"Resource\":[\"arn:aws:iam::$ACCOUNT_ID:role/resume-bot-execution\",
                    \"arn:aws:iam::$ACCOUNT_ID:role/resume-bot-task\"]}]}"
```

### 8.2 `.github/workflows/test.yml` — tests on every PR

```yaml
name: tests
on:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install uv && uv pip install --system -r requirements.txt pytest
      - run: pytest tests/ -v
```

### 8.3 `.github/workflows/deploy.yml` — test → build → push → deploy on main

```yaml
name: deploy
on:
  push:
    branches: [main]
permissions:
  id-token: write   # OIDC
  contents: read
env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: resume-bot
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install uv && uv pip install --system -r requirements.txt pytest
      - run: pytest tests/ -v
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::ACCOUNT_ID:role/resume-bot-github-deploy
          aws-region: ${{ env.AWS_REGION }}
      - uses: aws-actions/amazon-ecr-login@v2
        id: ecr
      - name: Build and push
        run: |
          IMAGE=${{ steps.ecr.outputs.registry }}/$ECR_REPOSITORY:${{ github.sha }}
          docker build -t $IMAGE .
          docker push $IMAGE
          echo "IMAGE=$IMAGE" >> $GITHUB_ENV
      - name: Render task definition
        id: taskdef
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: infra/task-definition.json
          container-name: resume-bot
          image: ${{ env.IMAGE }}
      - name: Deploy
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition: ${{ steps.taskdef.outputs.task-definition }}
          cluster: resume-bot
          service: resume-bot
          wait-for-service-stability: true
```

(GitHub's `ubuntu-latest` runners are x86, so plain `docker build` produces
the right architecture in CI — the `buildx --platform` flag is only needed on
your Mac.)

Branch protection: GitHub repo → Settings → Branches → require the `tests`
check to pass before merging to `main`.

---

## Order of operations checklist

1. Phase 0 repo fixes; verify `docker run` works locally end to end
2. AWS account + MFA + admin user + CLI + **budget alerts**
3. Anthropic Console spend limit
4. S3 bucket, sync `data/` + `vault/` (+ GitHub stats snapshot)
5. ECR repo, buildx push first image
6. SSM parameter with API key
7. IAM execution + task roles
8. Cluster, log group, security group, task definition, service → hit the IP
9. Bedrock guardrail + `screen_input()` + rate limiting in the app
10. OIDC provider + deploy role + the two workflows; merge a PR and watch it ship

## Known limitations of this setup (fine for testing)

- **HTTP only, IP changes each deploy.** Browsers will flag "not secure";
  Streamlit works fine over HTTP but anything you type is unencrypted in
  transit. The upgrade is an ALB + ACM cert + a $12/yr domain, ~+$18/mo.
- **One task = brief downtime during deploys** (old task stops, new one
  starts) and no horizontal scaling.
- **No WAF/CDN** — the app-layer rate limiting and Anthropic spend cap are
  your abuse protection.
