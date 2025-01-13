#!/bin/bash

set -e  # Exit on error

# Load environment variables
source deployment/scripts/load-env.sh
load_env

# Constants
MAX_LENGTH=20
TRIMMED_NARRATIVE_APP_NAME="${NARRATIVE_APP_NAME:0:$MAX_LENGTH}"
VPC_ID="vpc-06739f6c62b9b0da7"
NARRATIVE_ALB_ARN="arn:aws:elasticloadbalancing:eu-north-1:533267025675:loadbalancer/app/backend-chatbot-alb/14d33b15f13e7336"
NARRATIVE_LISTENER_ARN="arn:aws:elasticloadbalancing:eu-north-1:533267025675:listener/app/backend-chatbot-alb/14d33b15f13e7336/c57f6cce1504a047"

echo "Creating new target group in chatbot VPC..."
NEW_NARRATIVE_TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
    --name "${TRIMMED_NARRATIVE_APP_NAME}-tg-new" \
    --protocol HTTP \
    --port 8000 \
    --vpc-id ${VPC_ID} \
    --target-type ip \
    --health-check-path "/docs" \
    --health-check-interval-seconds 30 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --health-check-timeout-seconds 5 \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text)

echo "New target group created: ${NEW_NARRATIVE_TARGET_GROUP_ARN}"

# Create/Update listener rule to associate target group with ALB
echo "Creating listener rule..."

# Get all existing priorities
echo "Getting existing rule priorities..."
PRIORITIES=$(aws elbv2 describe-rules \
    --listener-arn "${NARRATIVE_LISTENER_ARN}" \
    --query 'Rules[*].Priority' \
    --output text)

# Find the next available priority
NEW_PRIORITY=1
for p in $(echo $PRIORITIES | tr ' ' '\n' | sort -n); do
    if [ "$p" != "default" ] && [ $NEW_PRIORITY -eq $p ]; then
        NEW_PRIORITY=$((p + 1))
    fi
done

echo "Using priority: ${NEW_PRIORITY}"

aws elbv2 create-rule \
    --listener-arn "${NARRATIVE_LISTENER_ARN}" \
    --priority ${NEW_PRIORITY} \
    --conditions "[{\"Field\":\"path-pattern\",\"Values\":[\"/${NARRATIVE_APP_NAME}/*\"]}]" \
    --actions "[{\"Type\":\"forward\",\"TargetGroupArn\":\"${NEW_NARRATIVE_TARGET_GROUP_ARN}\"}]" \
    --output text \
    --query 'Rules[0].RuleArn' || echo "Rule creation completed"

echo "Updating ECS service..."
aws ecs update-service \
    --cluster "${NARRATIVE_CLUSTER_NAME}" \
    --service "${NARRATIVE_SERVICE_NAME}" \
    --task-definition "${NARRATIVE_APP_NAME}" \
    --force-new-deployment \
    --network-configuration "{\"awsvpcConfiguration\":{\"subnets\":[\"${VPC_SUBNET_1}\",\"${VPC_SUBNET_2}\"],\"securityGroups\":[\"${SECURITY_GROUP}\"],\"assignPublicIp\":\"ENABLED\"}}" \
    --load-balancers "[{\"targetGroupArn\":\"${NEW_NARRATIVE_TARGET_GROUP_ARN}\",\"containerName\":\"${NARRATIVE_APP_NAME}\",\"containerPort\":8000}]" \
    --health-check-grace-period-seconds 120

# Update .env file with new ARNs
echo "Updating .env file with new ARNs..."
sed -i.bak "s|NARRATIVE_TARGET_GROUP_ARN=.*|NARRATIVE_TARGET_GROUP_ARN=\"${NEW_NARRATIVE_TARGET_GROUP_ARN}\"|" .env
sed -i.bak "s|NARRATIVE_ALB_ARN=.*|NARRATIVE_ALB_ARN=\"${NARRATIVE_ALB_ARN}\"|" .env
sed -i.bak "s|NARRATIVE_LISTENER_ARN=.*|NARRATIVE_LISTENER_ARN=\"${NARRATIVE_LISTENER_ARN}\"|" .env

echo "Migration complete! New configuration:"
echo "New target group ARN: ${NEW_NARRATIVE_TARGET_GROUP_ARN}"
echo "Using ALB: ${NARRATIVE_ALB_ARN}"
echo "Using Listener: ${NARRATIVE_LISTENER_ARN}"
echo ""
echo "Next steps:"
echo "1. Update your FastAPI application code to handle the new path prefix /${NARRATIVE_APP_NAME}/*"
echo "2. Monitor the ECS service status in AWS console"
echo "3. Test your application using the new URL path"