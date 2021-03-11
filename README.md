# Dockerizing image classification web app (Tf2 + Flask)

There is 3 basic part of this repo:

1. Train an Image classifier using TensorFlow 2 
2. Integrate the classifier in a flask app 
3. Dockerized the web app

But my main purpose was to learn how to use docker.

## Train a Image classifier using TensorFlow 2

I use transfer learning from pre-trained InceptionV3, made the last 15 layers trainable, and add a few layers on top of it. I do not perform that much image processing or data analysis as it is not the main purpose of this project.

But I use an Image-data-generator for some basic image augmentations. Also, I use <strong>flow_from_directory</strong> which do not load the full dataset at a time but load batch of data, augment it and feed to the network.

<strong>NOTE:</strong> One important thing worth to mention, <strong>"ImageDataGenerator"</strong> does not produce more data for the training, but replaces the original image batch with the augmented/Randomly transformed data.

I train the model for just 50 epochs. Here is how the training graph looks like:

trainging plot png

## Integrate the classifier in a flask web app

I am not a professional web developer, so my web-Ui may not be that much but it full fills my purpose.

Here is the initial UI.

initial ui png

Upload an image and click the submit button. The input image and the output label will be displayed under the input form. The model may not work all the time as it is not trained in a proper way. Here is how the page will look like after the prediction.

predict ui png

## Dockerize the web app

Now before we deep dive into docker installation and how it works, I think a little bit of knowledge about docker is necessary. The most Most important question is, What is docker and why should we use it?

In a simple word, DOCKER is the solution to <strong>"nothing is working in my system"</strong>. As a software developer (whether you are a machine learning engineer or not) we face this problem quite often that everything is running on my machine but nothing is working in the client machine. The main reason for is this is dependency mismatch. Docker solves this problem by standardizing the environment. By standardizing the environment I mean, putting all the files along with the required dependency and wrap the full process in an isolated box/container (docker container). In this way, our software becomes deployable in any environment (Windows/Linux/macOS).

So three main features of a dockerized project are:
1. It is highly portable
2. Isolate each project from another project and
3. it standardized the deploying environment and solve the problem <strong>"nothing is working in my system"</strong>.

By the way, a Virtual machine can perform the same thing for us. But one advantage of using Docker over Virtual machine is, Virtual machine is not expendable in terms of resource sharing. You have to allocate a fixed amount of memory for a Virtual machine. But in a Docker image, you do not have to do it. Also, a Virtual machine is not that portable while a docker image/docker container is highly portable.

This was to most complicated part to me. Most of the peoples like me find difficulties to install Docker in windows or Linex(Ubuntu). Here is how i did it:

### Windows Installations:
Go to this link (https://docs.docker.com/docker-for-windows/install/) and download it from Docker Hub. To install the Docker you much have a Windows 10 64-bit: Pro, Enterprise, or Education version. without this, it won't work. So, if your system did not full fill all the requirements, then you have to install the docket from the legacy versions. You can get all the legacy releases here (https://github.com/docker/toolbox/releases). From here you have to check all the releases from the top (download the .exe file and simply install it). My system full-fill all the requirements, so I did not try this process.

### Linux(Ubuntu) Installations:

1. First, add the Docker Engine repository’s key and address to apt’s repository index - 
>curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

2. Update package index and install the Docker engine:
>sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io

If these two commands run successfully, that means the docker is installed successfully. To check the installation you can run the hello world docker image from the docker hub. Run the comand bellow:
>sudo docker run hello-world

Now here I assume you already have an Nvidia GPU, GPU driver, and CUDA installed. If not please check on the internet how to do that.

3. Add NVIDIA Container Toolkit key and address to apt:
>distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

4. Install NVIDIA Container Toolkit:
>sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

5. Lastly restart the docker to finish it.
>sudo systemctl restart docker

Once everyting is installed, you can check all the docker containers you created so far by typing: 
>sudo docker ps -a

After completing all the steps above (training an image classifier, integrate the classifier in a web interface and install the docker) you should have something like this:

<pre>
image_classifire_root
	dataset/
	ouput/
	templates/
	test-data/
	uploads/
	main.py
	train_model.py
	predict.py
</pre>

<pre>
image_classifire_root - is the folder where everything is located.
dataset - contains the dataset
templates - contain the .html files
test-data - data you want to test in the classifier
uploads - contain all the images uploaded from the web UI
</pre>

All the folders are optional except "templates" as flask by default search the .html file in a templates folder.

<strong>NOTE:</strong> if you want to train the image classifier like me (using flow-from-directory), the structure of the dataset folder should be similar to me as it is the expected data directory by TensorFlow.

After everyting is being done, now we will create our Dockerfile. There are few basic commands we have to write in this file:

<pre>
1. "From"
	"From" describe which docker image system will fetch from the docker hub. As I am using Tensorflow 2, I fetch the latest version from the docker hub. The command is:
		FROM tensorflow/tensorflow:latest-gpu-py3

this command will download the TensorFlow-GPU version with python 3 from the docker hub.

2. "Copy"
	"Copy" tells us from where the files should be copied, and where to save in the newly downloaded docker container (downloaded using the FROM command). I created the Dockerfile in the same directory as all my other files. and I like to copy all the files in "/usr/app/" directory in the newly downloaded docker container. So the command will be:
		COPY . /usr/app/

"." means copy all the files from the present directory. This command copy all the files from the present directory to "/usr/app/" directory.

3. EXPOSE
	Dedicate a port for running this web app. The command is:
		EXPOSE 5000


4. WORKDIR
	The directory from where the full web app will run. As I copy all the files in "/usr/app/" directory, so I initialize this directory as my working directory.

5. RUN
	Run is used for installing the requirements. I wrote all the required dependencies with the respective version in a "requirements.txt" file. So the command is:
		RUN pip install -r requirements.txt

6. CMD
	This command said from which script the whole process will run. this is the same command we regularly use "python file_name.py".So the command is:
		CMD python main.py

"main.py" is the starting file for the whole program.
</pre>

After writing all the comands in the "Dockerfile", next we will build the docker image for our web app. Do that the command is: >sudo DOCKER_BUILDKIT=1 docker build -t flower_api .

Here "flower_api" is the name i choose for my docker container and "." means the present directory.

This command will read the "Dockerfile", execuat all the commands in it and create a docker image for us.

To run the docker image simpelly run:
>sudo docker run -p 5000:5000 flower_api

Here "flower_api" is the name you provide for your program.
