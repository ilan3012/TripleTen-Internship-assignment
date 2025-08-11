# TripleTen-Internship-assignment
## how to use
clone this repository  
`git clone https://github.com/ilan3012/TripleTen-Internship-assignment.git`  
navigate to the project folder  
`cd TripleTen-Internship-assignment/computer_use_demo`  
build the docker image  
`docker build -t image-computer-use-demo -f Dockerfile .`  
set up your api key as an environment variable  
`export NEBULS_API_KEY=%your_api_key%`  
`docker run -e NEBULS_API_KEY=$NEBULS_API_KEY -v $HOME/.anthropic:/home/computeruse/.anthropic -p 5900:5900 -p 8501:8501 -p 6080:6080 -p 8080:8080 -it image-computer-use-demo`  
