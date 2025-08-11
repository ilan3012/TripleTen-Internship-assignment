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
I looked at the code and identified where and how is the AI used: how does it gets the user messages, how its output is processed, what functions were used, etc. I also looked at anthropic api to better understand the code.
I then looked at Nebius API and searched what were the equivalent functions and compared their differences like the way they get input and produce output, and what features were available in one api but were lacking in the other one. 
I figured out what where the "deepest" part of the code(the parts which many of the other parts depended on) and started replacing the functions and other pieces of the code and moved "up" to the "surface"(the GUI).

Challenges: One of the main challenges were Docker. I was not that familiar with it and I had some trouble making it work. I figured it out in the end, but it was very time consuming.
Another challenge was that the code had a lot of buttons and features but in fact, a lot of them were not really implemented in the code. This had the effect of bloating the code and making it harder to understand what were the important pieces of code that I needed to pay attention to.
I also struggled with making the AI use the tools the correct way.
## Evaluation
*Accuracy: How accurate was the information the AI gave to the user
*Locating bugs: how well was the AI able to identify the problem
*Fixing bugs: how well was the AI able to fix the problem(after it had found it)


