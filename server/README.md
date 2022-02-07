# cGAN-RPS Server
Flask server application which listens for incoming requests and returns responses which contain of a base64 string which can be decoded to an image with a hand depicting rock, or hand, or scissors.
The server uses a cGAN-RPS model trained on the ```docs/HYPERPARAM_125.md``` (Attempt #2) specification, but on a 70% training-test split.

## Endpoints
1. GET ```/api/v1/``` -> returns a default hello response from the server
2. GET ```/api/v1/generate?target=<string>``` -> returns output from the cGAN model conditioned on <target> (eg: rock, paper, scissors) as image data encoded in base64 

# DevOps
## CI/CD
CI/CD is implemented through GitHub Actions and automates testing the app & deploying to Heroku.

The app is containerized in a Docker image and it will be deployed using the Heroku Container Registry.

### Progress
[x] CI (~test harness~, ~~job step~~) 

[x] CD (~~Heroku config~~, ~~Docker image~~, ~~job step~~) 

### Docker
Docker needs to be installed on the machine where the Flask app will be run.
To manually run the Docker app, run these commands in root repository folder :

1. docker build --tag cgan-rps-docker .
2. docker images 
3. docker run -d -p 5050:5050 cgan-rps-docker
4. docker ps
5. docker stop <app_id_from_step_4>
6. docker system prune

### CI
The continuous integration workflow can be found in ```.github/workflows/CI.yml``` and runs the test harness.

### CD
The continuous deployment workflow can be found in ```.github/workflows/CD.yml``` and runs the deployment commands to deploy the app to Heroku.
