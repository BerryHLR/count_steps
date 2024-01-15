# Project Overview
This project started at the beginning of March and was completed at the end of April, which lasted two months. The main part is to design a program that can count steps, based on TensorFlow's pose detection model. Additionally, the number of steps of each user should be recorded and ranked, and displayed on the front end in the form of a leaderboard.

这个项目从3月初开始，四月底完成，历时两个月。主要任务是设计一个可以计算步数的程序，以TensorFlow的人体检测模型为基础。同时，还要对每位用户的步数进行记录和排名，以排行榜的形式展示在前端。
# Procedures 
## Step1: Learning 
In this step, I learned the principle of PoseNet and Body track. Besides, I tried to run the pose detection model provided on the MoveNet official website and had basic ideas of the purpose of each part of the code.
## Step2: Designing 
In this step, after I had estimated the feasibility of identifying a step and distinguishing left and right, I set up a program framework based on the pose detection code and marked the objective for each part.
## Step3: Adjusting
In this step, I figured out how to apply a human gesture model to a real-time camera. Initially, I used the distance between the abscissa of the left and right feet as the threshold, but I just realized that it only worked when people walked sideways. So I changed it to the distance between two feet key points, which actually worked better. And then, I adjusted the parameter of the distance constantly and repeatedly to improve the accuracy of identifying a step.
## Step4: Running
In this step, after the program had successfully run in Google Colab,  I put the program into the local interpreter--VS Code, which required some libraries installed. Plus, I borrowed an external camera and connected it to my computer to better simulate real application scenarios. 
## Step5: Connecting
In this step, Sam gave me a simple example to show how to connect my program to the API URL Path. I made some changes to the example and successfully built a bridge between the code and the website, where the steps were shown.

# Challenges
## 1
The official website provides two examples: an image and a GIF, but what I want is to deal with the live camera. After searching online, the information obtained by the camera is a video stream, which is actually a frame-by-frame image, so the overall idea is to cut the video stream into an image and do human body posture recognition to count the number of steps
网上提供的示例一个是图像，一个是gif，但我们需要的是从摄像机实时反馈。经过查阅资料，摄像头拍摄取得的信息是一个视频流，实质上是一帧一帧的图像，所以整体思路就是把视频流切成一张一张照片分别进行人体姿态识别从而计算步数
## 2
Another issue that needs to be considered is how to prevent a person from constantly sticking out a leg to count step cheating. So I need to add a state variable, mark the left and right, and only when the state this time is not same as the last time can the number of steps be counted.
还有一个需要考虑的问题是如何避免一个人不停地伸出一条腿来count step作弊。所以要加入一个state变量，标记左右，只有当这次的state和上次不等的时候才能计算步数。
## 3
The difference between Google Colab and VS Code causes some problems. Firstly, a lot of libraries need to be downloaded, and the procedure is not as smooth as I thought., such as downloading but not finding in VS Code. I looked for the solution online, like restoring pip or using Anaconda, which actually didn’t work, and finally, I added two paths so the system could find this library.
Additionally, the code “take_photo” in Google Colab that opens the camera to capture images is not compatible with VS Code, which requires “cv2.VideoCapture()” and “cv2.imshow”. So the video format changes and some adjustments for visualizing the prediction of the image are needed.
Another example is that on Google Colab, it is necessary to convert the video into a byte stream and then visualize it, but this step is not required on VS Code.
Google Colab 和 VS Code 的不同造成了一些问题。
首先是有很多library需要下载，期间遇到很多困难比如已经下载了却在vs code里不能运行。我通过上网查阅解决方案，添加了两个path才让系统能找到这个库
另外，打开摄像头捕捉图像的程序也与vs code不适配。
再比如在Colab上面需要将视频转换成字节流形式再visualize, 而在vs code并不需要这一步。

# Perception and inspiration
This project lasted for half a semester, during which I grew from knowing little about human gesture recognition to understanding the principles and purposes of every step of the program. I also learned the ability to handle program errors. At first, I would become anxious and nervous when encountering errors, but now I hold the attitude of "encountering problems, solving problems" and can face them calmly. Through this period of hard work, my patience and attention to detail have been greatly improved, and my communication and time management skills have been exercised. 
Meanwhile, I have realized my shortcomings, such as insufficient programming experience and lack of understanding of functions used in models, which require much time to learn. In addition, I am not familiar with video and image processing, such as not knowing why images are required to be expanded in dimension initially, and need to learn and explore further.
Although I have learned Python and C++, I have only stayed at the level of understanding and have not really done a project. In class, we learn and practice in a modular way, but a real project combines what we have and have not learned, and requires a proficient mastery of each module and comprehensive ability. Therefore, I realize that learning programming cannot only stay in the modules of the class, but should be applied to practical problems to understand and master it truly.
In addition, my most profound understanding is that code is just a language, and it is more important to discover, analyze, and solve problems when writing programs. Whether doing things or writing programs, we must have a clear mind and purpose in order to achieve results with less effort.
Furthermore, when I encounter problems, besides researching on my own or searching for solutions online, I also consult with senior students who have relevant experience. I am deeply touched by their kindness and patience in answering my questions, and even offering to meet me in person to operate my computer to help me solve the restoration problem. Their kindness is extremely warm for me, who is stuck with difficult problems, and it deeply moves me.
Finally, I would like to sincerely thank the project leaders Leo WONG and Sam CHU, firstly for giving me this valuable opportunity to learn about human pose detection and accumulate relevant experience, and secondly, for their support and encouragement throughout this project. In the beginning, I was always nervous when attending project meetings, afraid that my report would not be satisfactory. But to my surprise and delight, they praised my achievements every time, and patiently explained to me when I was not very familiar with the technology or specialist vocabulary. It is my luck to be in such a warm, harmonious, and relaxed team.

此项目历时半个学期，我从最开始对人体识别的一无所知到最后可以清楚的知道每一步代码的原理和目的，
还学到了处理程序报错的能力，最开始一遇到error就会着急焦虑，现在秉持着“遇到问题，解决问题”的态度，已经可以坦然面对，
通过这段时间的努力，我的耐心、细心程度得到了极大的提高，沟通和时间管理能力都得到了锻炼。同时也意识到了自己的不足之处，比如编程经验不足，对于很多模型中用到的函数不够了解，需要花费大量时间学习。另外，我对于对视频和图片的处理也不熟悉，比如最初不知道为什么要expand images’ dimension，还需要加以学习和钻研
尽管我学过python和C++，但也仅仅停留在了解的层面，没有真正实操过项目。课堂是按模块式学习和练习，而一个真正的项目却是把学过的，没学过的集合在一起，更需要对各模块内容的熟练掌握和综合能力。因此我认识到学习编程不能只停留在课堂的模块，而是要在实际问题中运用，这样才能真正理解和掌握。
此外，我最深刻的认识是代码只是一种语言，编写程序更重要的是发现问题，分析问题和解决问题。不论是做事还是写程序，一定要有清晰的思路，有目的性的做事，才能事半功倍。
另外，当我遇到问题，除了自己研究或上网寻找解决方案，我还会向有相关经验的学长学姐咨询，令我感动的是他们都很耐心的解答我的问题，甚至主动提出meet me in person来操作我的电脑帮我处理问题。他们的kindness对于被棘手问题难住的我来说温暖无比，让我深深感动。
最后，我要感谢这个项目的负责人Leo WONG 和 Sam CHU. 一是要感谢他们给我这个宝贵的机会来学习人体姿态检测的知识，积累相关经验；二是要感谢他们在这个项目中一直以来对我的支持和鼓励。最开始我每次都怀着忐忑紧张的心情去参加project meeting，害怕自己的成果不尽如人意。但很让我惊喜和感动的是每次他们都对我的成果赞扬，而且当我有对于技术方面不太懂的时候也会耐心地给我讲解。能在这样温暖，和谐，轻松的团队是我的幸运。
