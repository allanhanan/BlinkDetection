#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

int main(){
    //check if camera available
    cv::VideoCapture camera(-1);
    if (!camera.isOpened()){
        cerr<<"Error: cannot open camera"<< endl;
        return 1;
    } 

    //load cascades
    cv::CascadeClassifier eyeCascade;
    eyeCascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye.xml");

    cv::CascadeClassifier faceCascade;
    faceCascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");

    cv::namedWindow("Webcam");
    cv:: Mat frame;     //cam input
    cv:: Mat gray;      //gray input

    //blink detection logic variables
    bool prevEyesDetected = false;//look at code to understand(idk how explain)
    int blinkCount = 0; //no of blinks (no use for it rn)
    int frameCount = 0; //just for threasholding
    const int minBlinkDuration = 5; //min no of frames with no eye to count as blink
    

    //main loop
    while (true){
        camera >> frame;
        if(frame.empty()){      //lil check
            cerr<<"Error: No frame captured"<<endl;
            break;
        }
        cvtColor(frame, gray, cv::COLOR_BGR2GRAY); //convert to gray

        //detect faces
        vector<cv::Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

        //detection loop
        for (const cv::Rect& face : faces) {
            //display face 
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0),2);

            // get face region
            cv::Mat faceReg = gray(face);

            //detect eyes within face
            vector<cv::Rect> eyes;
            eyeCascade.detectMultiScale(faceReg, eyes, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30,30));
            
            
            if(!eyes.empty()){
                //eyes detected
                prevEyesDetected = true;
                frameCount = 0; // Reset frame count since eyes are detected
            }else{
                //eyes not detected (ofc)
                if (prevEyesDetected){
                    frameCount++;
                    if (frameCount>=minBlinkDuration){
                        //blink successful
                        blinkCount++;
                        cout<<"Blink detected"<<endl;
                        //reset flags/variables
                        frameCount = 0;
                        prevEyesDetected = false;
                    }
                }
            }
            // for drawing eyes
            for (const cv::Rect& eye : eyes) {
                // convert relative eye coords(to face) to global coords to draw in "frame"
                cv::Point eyeCenter(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);
                int radius = cvRound((eye.width + eye.height) * 0.25);

                // draw circle
                circle(frame, eyeCenter, radius, cv::Scalar(0, 0, 255), 2);
            }
        }


        //display stuff
        cv::imshow("Webcam", frame);
        //breaks without this smh
        cv::waitKey(20);
    }
}