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
go to http://localhost:8080/ to see the app

## Adaptation Process
I looked at the code and identified where and how is the AI used: how does it gets the user messages, how is it's output procced, what functions were being used, etc. i also looked at anthropic api to better understand the code.
i then looked at Nebius API and searched what were the equivlent functions and coompred their diffrence like the way the get input and prodoce output, and what features were avilable in one api but were lacking in the other one. 
i figured out what where the "deapest" part of the code(the parts which many of the other parts deapnd on) and started replacing the functions and other pieces of the code and moved "up" to the "surface"(the GUI).


## Evaluation
