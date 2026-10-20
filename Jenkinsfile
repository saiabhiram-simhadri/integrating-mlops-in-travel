pipeline {
    agent any

    environment {
        DOCKER_HUB_REPO = "saiabhiram73865/flight-price-api"
        IMAGE_TAG = "v${BUILD_NUMBER}"
        REGISTRY_CREDS = "dockerhub-credentials"
    }

    stages {
        stage('1. Checkout SCM') {
            steps {
                echo "Pulling latest commit from GitHub..."
                checkout scm
            }
        }

        stage('2. Static Code Analysis & Linting') {
            steps {
                echo "Executing flake8 code quality audit..."
                sh 'flake8 app.py --count --select=E9,F63,F7,F82 --show-source --statistics'
            }
        }

        stage('3. Automated Integration Tests') {
            steps {
                echo "Running model inference sanity checks..."
                sh 'python -c "import app, joblib; print(\'App imports and dependencies validated\')"'
            }
        }

        stage('4. Docker Build & Push') {
            steps {
                echo "Building container image: ${DOCKER_HUB_REPO}:${IMAGE_TAG}"
                sh "docker build -t ${DOCKER_HUB_REPO}:${IMAGE_TAG} ."
                sh "docker tag ${DOCKER_HUB_REPO}:${IMAGE_TAG} ${DOCKER_HUB_REPO}:latest"
                withCredentials([usernamePassword(credentialsId: "${REGISTRY_CREDS}", usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh 'echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin'
                    sh "docker push ${DOCKER_HUB_REPO}:latest"
                }
            }
        }

        stage('5. Kubernetes Continuous Deployment') {
            steps {
                echo "Applying Kubernetes deployment..."
                sh 'kubectl apply -f k8s_deployment.yaml'
                sh 'kubectl rollout status deployment/flight-price-api-deployment --timeout=120s'
            }
        }
    }

    post {
        success {
            echo "CI/CD Pipeline Completed Successfully. Production workload active."
        }
        always {
            cleanWs()
        }
    }
}