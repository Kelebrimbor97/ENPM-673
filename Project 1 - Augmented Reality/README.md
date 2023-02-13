# Project 1: Augmented-Reality

## Project Description

In this project, we were provided with a video that had a fiducial marker. We were tasked with imposing an image and a cube onto this tag. The approach for doin this is detailed in the report.

## Running instructions

This is my submission for the first project for the Subject ENPM 673 for the SPring 2022 semester.

Part 1a,1b and 2a:

The code for parts 1a, 1b, and 2a is titled 'prj1_q1a.py'. For Running this code, it is advised (not necessary) to use an IDE such as visual studio.

1. In your working directory (preferably the one where the code is located), ensure that you have the video file - '1tagvideo.mp4' and the image file - 'testudo.png'
2. Run the code and get the results
3. By default, sections of the code have been commented to prevent the writing of the video to storage of the host computer. Lines 39, 90, and 104 may be uncommented if user wants to write the video to storage.
4. Lines 66 to 79 may be uncommented if user wants to visualise ordered corners.

The code for part 2b is titled 'cube.py' and runs in the exact same way, without the need for the 'testudo.png' image.

Note: If the code is cloned, please ensure that you check the paths in the codes as input data has not been Uploaded. To view the video click the GIFs to be redirected to their youtube uploads.

### 1. Morphological Operations

Basic morphological operations are performed on the video to isolate the QR code. The output for which is what we see below.

[![Morphological_Image_processing_AdobeExpress](https://user-images.githubusercontent.com/35636842/218368728-35bad771-17fc-4388-a5d9-7d82de198b73.gif)](https://youtu.be/K2lSg51COJ0)

### 2. Ordered Corner Detection

Now that we have the outline of the sqare of the QR code, we need to find the corners in order. This is important to get the orientation of the QR code.

[![ezgif com-optimize](https://user-images.githubusercontent.com/35636842/218370397-a4d85010-2d91-48cc-bdc0-b21f8113ec73.gif)](https://youtu.be/vW_rvp5WV1s)

### 3. Image Imposition on tag

Finally, we impose the image onto the marker.

[![Testudo_Imposition_on_Fiducial_Marker](https://user-images.githubusercontent.com/35636842/218371698-48caacd1-4087-48ff-880e-47d440ba41d6.gif)](https://youtu.be/-3TvXzFVbGk)

### 4. Cube Imposition on tag

The next step is to 'place' an AR cube onto the marker.

[![AR_Cube](https://user-images.githubusercontent.com/35636842/218374017-6fa83d84-e12b-4b27-9abc-ddf9ce52e645.gif)](https://youtu.be/iCmC9NynMJg)

### 5. BONUS! Blooper Reel

A funny blooper.

[![Blooper_Reel_AdobeExpress](https://user-images.githubusercontent.com/35636842/218374658-0c686d49-c87a-4c22-b574-f66b8b5601c1.gif)](https://youtu.be/cD1I4kXAkUM)
