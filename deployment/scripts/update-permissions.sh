#!/bin/bash

# Load environment variables
source deployment/scripts/load-env.sh
load_env

# Create policy for CloudWatch Logs
aws iam put-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-name CloudWatchLogsPolicy \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "*"
            }
        ]
    }'

# Update security group
aws ec2 authorize-security-group-ingress \
    --group-id ${SECURITY_GROUP} \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

# Update service with target group 
aws ecs update-service \
    --cluster ${NARRATIVE_CLUSTER_NAME} \
    --service ${NARRATIVE_SERVICE_NAME} \
    --load-balancers "targetGroupArn=${NARRATIVE_TARGET_GROUP_ARN},containerName=${NARRATIVE_APP_NAME},containerPort=8000"