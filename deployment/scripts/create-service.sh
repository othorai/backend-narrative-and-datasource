#!/bin/bash

set -e  # Exit on error

# Load environment variables
source deployment/scripts/load-env.sh
load_env

echo "Creating ECS service..."
aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition $APP_NAME \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$VPC_SUBNET_1,$VPC_SUBNET_2],securityGroups=[$SECURITY_GROUP],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=$TARGET_GROUP_ARN,containerName=$APP_NAME,containerPort=8000" \
    --health-check-grace-period-seconds 120 \
    --region eu-north-1

echo "Waiting for service to become stable..."
aws ecs wait services-stable \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region eu-north-1

echo "Service created successfully!"
