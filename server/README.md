# cGAN-RPS Server
Flask server application which listens for incoming requests and returns responses which contain of a base64 string which can be decoded to an image with a hand depicting rock, or hand, or scissors.

# CI/CD
CI/CD is implemented through GitHub Actions and automates testing the app & deploying to Heroku.
The app is containerized in a Docker image and it will be deployed using the Heroku Container Registry.

## Progress
[ ] CI (test harness, job step) 
[ ] CD (Heroku config, Docker image, job step) 

## CI
The continuous integration workflow can be found in .github/workflows/CI.yml and runs the test harness.

## CD
The continuous deployment workflow can be found in .github/workflows/CD.yml and runs the deployment commands to deploy the app to Heroku.
