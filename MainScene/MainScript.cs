using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.Features2dModule;
using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.PhotoModule;
using OpenCVForUnity.TrackingModule;
using UnityEngine.UIElements;
using System.Threading;
using System.Security.AccessControl;
using OpenCVForUnity.VideoModule;

public class MainScript : MonoBehaviour
{
    // All

    /// AppState<summary>
    /// 0 init
    /// 1 located
    /// 2 video
    /// 3 segment
    /// 4 Detect
    /// 5 warp
    /// 6 remove
    /// 7 pause
    /// </summary>
    public int AppState = 0;
    System.Diagnostics.Stopwatch stopWatch;

    // UI
    public UnityEngine.UI.Button localizeButton;
    public UnityEngine.UI.Button videoButton;
    public UnityEngine.UI.Button segmentButton;
    public UnityEngine.UI.Button detectButton;
    public UnityEngine.UI.Button captureButton;
    public UnityEngine.UI.Button warpButton;
    public UnityEngine.UI.Button removeButton;
    public UnityEngine.UI.Button saveButton;
    public UnityEngine.UI.Button clearButton;
    public RawImage imageView;
    public RawImage resultView;
    public Text LeftText;
    public Text RightText;

    // Location
    private float lantitude;
    private float longitude;
    public Position currentPosition;

    // Camera 
    public Camera defaultCamera;
    private WebCamTexture webCam;
    private int requestImageSize = 720;
    private int imageHeight;
    private int imageWidth;
    private Size srcSize;
    private Texture2D imageTexture;
    private Texture2D resultTexture;
    private Texture2D maskTexture;
    private Texture2D warpTexture;

    // remove
    public Mat libMat;
    public Mat inputMat;
    public Mat oldInputMat;
    public Mat maskMat;
    public Mat warpMat;
    public Mat resultMat;

    ORB detector;
    Mat emptyMask;
    Mat kernelMask;
    DescriptorMatcher matcher;
    List<MatOfDMatch> matchMatList; 
    DMatch[] matchTwo;
    Mat homo;

    MatOfKeyPoint libKeyPointsMat;
    Mat libDescriptor;
    KeyPoint[] libKeyPointsArray;
    List<Point> libPointsListHomo;
    MatOfPoint2f libKeyPointsMatHomo;
    MatOfPoint2f libKeyPointsMatHomoPre;

    MatOfKeyPoint inputKeyPointsMat;
    Mat inputDescriptor;
    KeyPoint[] inputKeyPointsArray;
    List<Point> inputPointsListHomo;
    MatOfPoint2f inputKeyPointsMatHomo;
    MatOfPoint2f inputKeyPointsMatHomoPre;



    // segment
    [SerializeField, FilePopup("*.tflite")]
    public string deeplabFileName = "deeplabv3_257_mv_gpu.tflite";
    //public string deeplabFileName = "deeplabv3_1_default_1.tflite";
    [SerializeField] 
    public ComputeShader compute = null;
    DeepLab deepLab;

    // detect
    [SerializeField, FilePopup("*.tflite")]
    public string ssdFileName = "coco_ssd_mobilenet_quant.tflite";
    SSD ssd;
    int lastNum = 0;
    SSD.Result[] ssdResults;
    SSD.Result[] ssdResultsOld = new SSD.Result[] { };
    ArrayList ssdResultArray = new ArrayList();
    ArrayList ssdResultArrayOld = new ArrayList();

    //Tracking
    MultiTracker trackers;
    MatOfRect2d objects;

    MatOfByte mMOBStatus;
    MatOfFloat mMOFerr;
    List<Point> newLibPoints = new List<Point>();
    List<Point> newInputPoints = new List<Point>();

    void Awake()
    {
        // All initialization
        AppState = 0;
        Screen.SetResolution(2400, 1080, false);
        stopWatch = new System.Diagnostics.Stopwatch();

        // UI initialization
        localizeButton.onClick.AddListener(LocalizeState);
        videoButton.onClick.AddListener(VideoState);
        segmentButton.onClick.AddListener(SegmentState);
        detectButton.onClick.AddListener(DetectState);
        warpButton.onClick.AddListener(WarpState);
        removeButton.onClick.AddListener(RemoveState);
        captureButton.onClick.AddListener(ShotImage);
        segmentButton.onClick.AddListener(SegmentState);
        saveButton.onClick.AddListener(SaveState);
        clearButton.onClick.AddListener(ClearState);
        //StartCoroutine(CallCamera());

        // Camera initialization
        WebCamDevice[] camDevices = WebCamTexture.devices;
        string deviceName = camDevices[0].name;
        webCam = new WebCamTexture(deviceName, requestImageSize, requestImageSize, 10);
        webCam.Play();
        imageHeight = webCam.height;
        imageWidth = webCam.width;
        //int angle = webCam.videoRotationAngle;
        //imageView.transform.Rotate(new Vector3(0, 0, -angle));
        //resultView.transform.Rotate(new Vector3(0, 0, -angle));
        srcSize = new Size(imageWidth, imageHeight);
        resultTexture = new Texture2D(webCam.width, webCam.height, TextureFormat.RGBA32, false);
        maskTexture = new Texture2D(webCam.width, webCam.height, TextureFormat.RGBA32, false);
        warpTexture = new Texture2D(webCam.width, webCam.height, TextureFormat.RGBA32, false);

        // Remove initialization
        libMat = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        inputMat = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        oldInputMat = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        maskMat = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        warpMat = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        resultMat = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        kernelMask = Imgproc.getStructuringElement(0, new Size(60, 60));
        matchMatList = new List<MatOfDMatch>();
        libPointsListHomo = new List<Point>();
        inputPointsListHomo = new List<Point>();

        libKeyPointsMatHomo = new MatOfPoint2f();
        libKeyPointsMatHomoPre = new MatOfPoint2f();
        inputKeyPointsMatHomo = new MatOfPoint2f();
        inputKeyPointsMatHomoPre = new MatOfPoint2f();
        emptyMask = new Mat();
        detector = ORB.create(2000);
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);
        //matchMatList = new MatOfDMatch();
        libKeyPointsMat = new MatOfKeyPoint();
        libDescriptor = new Mat();
        inputKeyPointsMat = new MatOfKeyPoint();
        inputDescriptor = new Mat();

        //segment initialization
        string path = Path.Combine(Application.streamingAssetsPath, deeplabFileName);
        deepLab = new DeepLab(path, compute);
        var deepLabResizeOptions = deepLab.ResizeOptions;
        deepLabResizeOptions.rotationDegree = 0;
        deepLabResizeOptions.flipX = false;
        deepLabResizeOptions.flipY = false;
        deepLab.ResizeOptions = deepLabResizeOptions;

        //detect initialization
        path = Path.Combine(Application.streamingAssetsPath, ssdFileName);
        ssd = new SSD(path);
        var ssdResizeOptions = ssd.ResizeOptions;
        ssdResizeOptions.rotationDegree = 0;
        ssdResizeOptions.flipX = false;
        ssdResizeOptions.flipY = false;
        ssd.ResizeOptions = ssdResizeOptions;

        //Tracking initialization
        trackers = MultiTracker.create ();
        objects = new MatOfRect2d ();

        //optical flow initialization
        mMOBStatus = new MatOfByte();
        mMOFerr = new MatOfFloat();
    }
    void Start()
    {
        // read lib image and detect
        StartCoroutine(LibProcess());
        //Utils.matToTexture2D(libMat, resultTexture);
        //resultView.texture = resultTexture;
    }

    void FixedUpdate()
    {
        /// AppState<summary>
        /// 0 init
        /// 1 located
        /// 2 video
        /// 3 segment
        /// 4 Detect
        /// 5 warp
        /// 6 remove
        /// 7 pause
        /// 8 tracking
        /// </summary>
        LeftText.text = "";
        RightText.text = "";
        if (AppState == 2)
        {
            DoLeftImage();
            DoVideo();
            DoRightImage(imageTexture);
            //RawImageView.canvasRenderer.SetTexture(webCam);
            //imageTexture.SetPixels(webCam.GetPixels());
            //imageTexture.Apply();
            //imageView.texture = imageTexture;
            //Utils.texture2DToMat(imageTexture, inputMat);
            //Utils.matToTexture2D(inputMat, resultTexture);
            //resultView.texture = resultTexture;
            //RawImageView.transform.Rotate(new Vector3(0, 0, angle));
            //Texture2D tNew = RotateTexture(t);
        }
        else if (AppState == 3)
        {
            DoLeftImage();
            DoSegment();
            DoRightImage(maskTexture);
            //imageTexture.SetPixels(webCam.GetPixels());
            //imageTexture.Apply();
            //imageView.texture = imageTexture;
            //resultView.texture = maskTexture;
            //RawImageView.transform.Rotate(new Vector3(0, 0, angle));
            //Texture2D tNew = RotateTexture(t);
        }
        else if (AppState == 4)
        {
            DoLeftImage();
            DoDetect();


            stopWatch.Start();
            Utils.texture2DToMat(imageTexture, inputMat);
            LeftText.text = "detected  " + imageHeight.ToString() + imageWidth.ToString() + "  ";
            for (int i = 0; i < ssdResults.Length; i++)
            {
                if (ssdResults[i].score > 0.5)
                {
                    LeftText.text += ssdResults[i].classID.ToString() + "  ";
                    LeftText.text += (int)(ssdResults[i].top * imageWidth) + " ";
                    LeftText.text += (int)(ssdResults[i].left * imageHeight) + " ";
                    LeftText.text += (int)(ssdResults[i].bottom * imageWidth) + " ";
                    LeftText.text += (int)(ssdResults[i].right * imageHeight) + " ";
                    Imgproc.rectangle(inputMat, new Point(ssdResults[i].left * imageWidth, imageHeight - ssdResults[i].top * imageHeight),
                        new Point(ssdResults[i].right * imageWidth, imageHeight - ssdResults[i].bottom * imageHeight),
                        new Scalar(0, 0, 200), 3);
                    Imgproc.putText(inputMat, ssdResults[i].classID.ToString(),
                        new Point(ssdResults[i].left * imageWidth + 5, imageHeight - ssdResults[i].top * imageWidth + 5),
                        Imgproc.FONT_HERSHEY_PLAIN, 3, new Scalar(10, 0, 0));
                }
            }
            //Imgproc.rectangle(inputMat, new Point(20, 40), new Point(40, 80), new Scalar(0, 0, 200), 3);
            stopWatch.Stop();
            RightText.text += "mask time = " + stopWatch.Elapsed.TotalMilliseconds;
            stopWatch.Reset();
            Utils.matToTexture2D(inputMat, maskTexture);
            DoRightImage(maskTexture);
            //var size = ((RectTransform)cameraView.transform).rect.size;
            //for (int i = 0; i < 10; i++)
            //{
            //    SetFrame(frames[i], results[i], size);
            //}
            //resultView.material = ssd.transformMat;
        }
        else if (AppState == 5)
        {
            DoLeftImage();
            Utils.texture2DToMat(imageTexture, inputMat);
            DoWarp();
            Utils.matToTexture2D(warpMat, warpTexture);
            DoRightImage(warpTexture);
            //resultView.texture = result;
            //Color32[] imagePixels = result.GetPixels32();
            //GCHandle imageHandle = GCHandle.Alloc(imagePixels, GCHandleType.Pinned);
            //Color32[] maskPixels = result.GetPixels32();
            //GCHandle maskHandle = GCHandle.Alloc(maskPixels, GCHandleType.Pinned);
            //IntPtr imagePointer = imageHandle.AddrOfPinnedObject();
            //IntPtr maskPointer = maskHandle.AddrOfPinnedObject();
        }
        else if (AppState == 6)
        {
            DoLeftImage();
            Utils.texture2DToMat(imageTexture, inputMat);
            Graphics.CopyTexture(imageTexture, resultTexture);
            DoRemove();
            //Utils.matToTexture2D(resultMat, resultTexture);
            DoRightImage(resultTexture);
        }
        else if (AppState == 7)
        {
            DoLeftImage();
            Utils.texture2DToMat(imageTexture, inputMat);
            Graphics.CopyTexture(imageTexture, resultTexture);
            DoRemove();
            //Utils.matToTexture2D(resultMat, resultTexture);
            DoRightImage(resultTexture);
            //DoLeftImage();
            //Utils.texture2DToMat(imageTexture, inputMat);
            //DoDetect();
            //trackers.clear();
            //trackers.add(TrackerKCF.create(), inputMat, new Rect2d(40, 60, 200, 300));
            //int find = 0;
            //for (int i = 0; i < ssdResults.Length; i++)
            //{
            //    if (ssdResults[i].score > 0.5)
            //    {
            //        find = 1;
            //        trackers.add(TrackerKCF.create(), inputMat,
            //            new Rect2d(ssdResults[i].left * imageWidth, ssdResults[i].bottom * imageHeight,
            //            (ssdResults[i].right - ssdResults[i].left) * imageWidth, (ssdResults[i].top - ssdResults[i].bottom) * imageHeight));
            //    }
            //}
            //DoRightImage(imageTexture);
            //if (find == 1) {
            //    AppState = 8;
            //}
            //DoDetect();
            //DoLeftImage();
            //Utils.texture2DToMat(imageTexture, inputMat);
            //try
            //{
            //    stopWatch.Start();
            //    trackers.update(inputMat, objects);
            //    stopWatch.Stop();
            //    RightText.text += "track time = " + stopWatch.Elapsed.TotalMilliseconds;
            //    stopWatch.Reset();

            //}
            //catch (Exception ex)
            //{
            //    RightText.text += "track" + ex.Message;
            //}
            //Rect2d[] objectsArray = objects.toArray();
            //for (int i = 0; i < objectsArray.Length; i++)
            //{
            //    Imgproc.rectangle(inputMat, objectsArray[i].tl(), objectsArray[i].br(), new Scalar(0, 0, 200), 2);
            //}
            //Utils.matToTexture2D(inputMat, resultTexture);
            //DoRightImage(resultTexture);
            //if (objectsArray.Length == 0)
            //    AppState = 7;
        }
        else if (AppState == 8)
        {
            try
            {
                DoLeftImage();
                stopWatch.Start();
                Utils.texture2DToMat(imageTexture, inputMat);
                inputMat.copyTo(resultMat);
                if(inputKeyPointsMatHomoPre.rows() < 20)
                {
                    Graphics.CopyTexture(imageTexture, resultTexture);
                    DoWarp();
                    LeftText.text += "new " + inputKeyPointsMatHomoPre.rows();
                    //Utils.matToTexture2D(warpMat, warpTexture);
                    //DoRightImage(warpTexture);
                    DoRightImage(imageTexture);

                    inputKeyPointsMatHomo.copyTo(inputKeyPointsMatHomoPre);
                    libKeyPointsMatHomo.copyTo(libKeyPointsMatHomoPre);
                    inputMat.copyTo(oldInputMat);
                }
                else
                {
                    Video.calcOpticalFlowPyrLK(oldInputMat, inputMat, inputKeyPointsMatHomoPre, inputKeyPointsMatHomo, mMOBStatus, mMOFerr);

                    if (mMOBStatus.rows () > 0) {
                        LeftText.text += "old " + inputKeyPointsMatHomo.rows() + " " + inputKeyPointsMatHomo.cols() + " ";
                        List<Point> libPointsPreList = libKeyPointsMatHomoPre.toList();
                        List<Point> inputPointsPreList = inputKeyPointsMatHomoPre.toList ();
                        List<Point> libPointsList = libKeyPointsMatHomo.toList ();
                        List<Point> inputPointsList = inputKeyPointsMatHomo.toList ();
                        List<byte> byteStatus = mMOBStatus.toList ();
                        newLibPoints.Clear();
                        newInputPoints.Clear();
                    
                        int x = 0;
                        int y = byteStatus.Count;
                                                        
                        for (x = 0; x < y; x++) {
                            if (byteStatus [x] == 1) {
                                Point ptLib = libPointsPreList[x];
                                Point ptInput = inputPointsList [x];
                                Point ptInputPre = inputPointsPreList [x];
                                newLibPoints.Add(ptLib);
                                newInputPoints.Add(ptInput);
                                Imgproc.circle(resultMat, ptInput, 5, new Scalar(255, 0, 0, 255), 2);
                                Imgproc.line(resultMat, ptInput, ptInputPre, new Scalar(255, 0, 0, 255), 3);
                            }
                        }
                    }
                    //Utils.matToTexture2D(resultMat, resultTexture);
                    //DoRightImage(resultTexture);

                    inputKeyPointsMatHomo.fromList(newInputPoints);
                    libKeyPointsMatHomo.fromList(newLibPoints);
                    LeftText.text += "for homo : " + newInputPoints.Count;
                    LeftText.text += "mat: " + inputKeyPointsMatHomo.rows() + " " + inputKeyPointsMatHomo.rows();
                    homo = Calib3d.findHomography(libKeyPointsMatHomo, inputKeyPointsMatHomo);
                    LeftText.text += homo.dims().ToString();
                    Imgproc.warpPerspective(libMat, warpMat, homo, inputMat.size());
                    Imgproc.cvtColor(warpMat, warpMat, Imgproc.COLOR_RGB2BGR);

                    Utils.matToTexture2D(warpMat, warpTexture);
                    DoRightImage(warpTexture);

                    inputMat.copyTo(oldInputMat);
                    inputKeyPointsMatHomo.copyTo(inputKeyPointsMatHomoPre);
                    libKeyPointsMatHomo.copyTo(libKeyPointsMatHomoPre);


                    //homo = Calib3d.findHomography(libKeyPointsMatHomo, inputKeyPointsMatHomo);
                    //LeftText.text += homo.dims().ToString();

                    //Imgproc.warpPerspective(libMat, warpMat, homo,inputMat.size());
                    //Imgproc.cvtColor(warpMat, warpMat, Imgproc.COLOR_RGB2BGR);

                    //stopWatch.Stop();
                    //RightText.text += "wrap time = " + stopWatch.Elapsed.TotalMilliseconds;
                    //stopWatch.Reset();
                }
                stopWatch.Stop();
                RightText.text += "opf time = " + stopWatch.Elapsed.TotalMilliseconds;
                stopWatch.Reset();
                //Utils.matToTexture2D(resultMat, resultTexture);

            }
            catch(Exception ex)
            {
                RightText.text += "opf " + ex.Message;
            }
        }
        else if (AppState == 9)
        {
            ;
        }
        LeftText.text = "";
        RightText.text = "";

    }

    public void LocalizeState()
    {
        Debug.Log("Localizing!");
        AppState = 1;
        RightText.text = "Localizing";

    }
    public void VideoState()
    {
        Debug.Log("Showing Video!");
        AppState = 2;
        RightText.text = "Showing Video";

    }
    public void SegmentState()
    {
        Debug.Log("Segmenting!");
        AppState = 3;
        RightText.text = "Segmenting";
    }
    public void DetectState()
    {
        Debug.Log("Detecting!");
        AppState = 4;
        RightText.text = "Detecting";
    }
    public void WarpState()
    {
        Debug.Log("Warping!");
        AppState = 5;
        RightText.text = "Warping";
    }
    public void RemoveState()
    {
        Debug.Log("Removing!");
        AppState = 6;
        RightText.text = "Removing";
        //Texture2D t0 = ImageView.texture as Texture2D;
        //Texture2D t1 = RotateN90(t0);
        //byte[] by = t1.EncodeToPNG();
        //File.WriteAllBytes(Application.persistentDataPath + "/input1.png", by);
        //inputTex = imageView.texture;
        //if(!inputTex)
        //    Debug.Log("input null");
        //deepLab.Invoke(inputTex);
        //Texture2D newTexture = deepLab.GetResultTexture2D() as Texture2D;
        ////newTexture.Resize(480, 640, TextureFormat.RGBA32, false);
        //byte[] byt = newTexture.EncodeToPNG();
        //File.WriteAllBytes(Application.persistentDataPath + "/mask0.png", byt);
        //newTexture = RotateP90(newTexture);
        //imageView.texture = newTexture;
    }
    public void SaveState()
    {
        Debug.Log("Saviing!");
        AppState = 7;
        RightText.text = "Saving";
    }
    public void ClearState()
    {
        Debug.Log("Saviing!");
        AppState = 8;
        RightText.text = "Saving";
    }
    public void DoLeftImage()
    {
        imageTexture.SetPixels(webCam.GetPixels());
        imageTexture.Apply();
        imageView.texture = imageTexture;
    }
    public void DoRightImage(Texture2D resultT)
    {
        resultView.texture = resultT;
    }
    public void DoVideo()
    {
        ;
    }
    public void DoSegment()
    {
        stopWatch.Start();
        deepLab.Invoke(webCam);
        maskTexture = deepLab.GetResultTexture2D() as Texture2D;
        stopWatch.Stop();
        RightText.text += "segment time = " + stopWatch.Elapsed.TotalMilliseconds;
        stopWatch.Reset();
    }
    public void DoDetect()
    {
        stopWatch.Start();
        ssd.Invoke(webCam);
        ssdResults = ssd.GetResultsNew();
        stopWatch.Stop();
        RightText.text += "detect time = " + stopWatch.Elapsed.TotalMilliseconds;
        stopWatch.Reset();
    }
    public void DoWarp()
    {
        try
        {
            stopWatch.Start();
            detector.detectAndCompute(inputMat, emptyMask, inputKeyPointsMat, inputDescriptor);
            inputKeyPointsArray = inputKeyPointsMat.toArray();
            LeftText.text = "all : " + inputKeyPointsArray.Length.ToString();
            matcher.knnMatch(inputDescriptor, libDescriptor, matchMatList, 2);

            inputPointsListHomo.Clear();
            libPointsListHomo.Clear();
            for(int i = 0;i<matchMatList.Count;i++)
            {
                matchTwo = matchMatList[i].toArray();
                if(matchTwo[0].distance < 0.5 * matchTwo[1].distance)
                {
                    inputPointsListHomo.Add(inputKeyPointsArray[matchTwo[0].queryIdx].pt);
                    libPointsListHomo.Add(libKeyPointsArray[matchTwo[0].trainIdx].pt);
                }
            }
            inputKeyPointsMatHomo.fromList(inputPointsListHomo);
            libKeyPointsMatHomo.fromList(libPointsListHomo);
            LeftText.text += "for homo : " + inputPointsListHomo.Count;
            stopWatch.Stop();
            RightText.text += "match time = " + stopWatch.Elapsed.TotalMilliseconds;
            stopWatch.Reset();
        }
        catch (Exception ex)
        {
            RightText.text += "  match" + ex.Message;
        }
        try
        {
            homo = Calib3d.findHomography(libKeyPointsMatHomo, inputKeyPointsMatHomo);
            LeftText.text += homo.dims().ToString();

            Imgproc.warpPerspective(libMat, warpMat, homo, inputMat.size());
            Imgproc.cvtColor(warpMat, warpMat, Imgproc.COLOR_RGB2BGR);

            stopWatch.Stop();
            RightText.text += "wrap time = " + stopWatch.Elapsed.TotalMilliseconds;
            stopWatch.Reset();

        }
        catch (Exception ex)
        {
            RightText.text += "  wrap " + ex.Message;
        }

    }

    public void DoRemove()
    {
        DoWarp();
        Utils.matToTexture2D(warpMat, warpTexture);
        DoDetect();
        //ssdResultArray.Clear();
        //for (int i = 0; i < ssdResults.Length; i++)
        //{
        //    if (ssdResults[i].classID == 0 && ssdResults[i].score > 0.5)
        //    {
        //        ssdResultArray.Add(ssdResults[i]);
        //    }
        //}
        //LeftText.text += "person : " + ssdResultArray.Count.ToString() + "  ";
        //if (ssdResultArray.Count < ssdResultArrayOld.Count)
        //    ssdResultArray = ssdResultArrayOld;
        //else
        //    ssdResultArrayOld = ssdResultArray;
        //if (ssdResults.Length <= 0)
        //    ssdResultsOld.CopyTo(ssdResults);
        //else
        //    ssdResults.CopyTo(ssdResultsOld);
        int personNum = 0;
        for (int i = 0; i < ssdResults.Length; i++)
        {
            if ((ssdResults[i].classID == 0) && ssdResults[i].score > 0.5)
            {
                personNum++;
            }
        }
        if(personNum == 0)
        {
            ssdResults = ssdResultsOld;
        }
        else
        {
            ssdResultsOld = ssdResults;
        }
        try
        {
            Imgproc.rectangle(maskMat, new Point(0,0),
                new Point(imageWidth-1, imageHeight-1),
                new Scalar(0, 0, 0), -1);
            LeftText.text = "";
            stopWatch.Start();
            inputMat.copyTo(resultMat);
            for (int i = 0; i < ssdResults.Length; i++)
            {
                if ((ssdResults[i].classID == 0 ) && ssdResults[i].score > 0.5)
                {
                    int leftX = (int)(ssdResults[i].left * imageWidth);
                    int rightX = (int)(ssdResults[i].right * imageWidth);
                    int bottomY = (int)(ssdResults[i].top * imageHeight);
                    int topY = (int)(ssdResults[i].bottom * imageHeight);
                    //Imgproc.rectangle(maskMat, 
                    //    new Point(0,0),
                    //    new Point(imageWidth-1, imageHeight-1),
                    //    new Scalar(0, 0, 0), -1);
                    //Imgproc.rectangle(maskMat, 
                    //    new Point(leftX, bottomY),
                    //    new Point(rightX, topY),
                    //    new Scalar(255, 255, 255), -1);
                    LeftText.text += ssdResults[i].classID.ToString() + "  ";
                    LeftText.text += leftX + " ";
                    LeftText.text += bottomY + " ";
                    LeftText.text += rightX + " ";
                    LeftText.text += topY + " ";
                    //warpMat.copyTo(resultMat, maskMat);
                    linearClone(leftX - (int)(0.2 * (rightX - leftX + 1)), bottomY - (int)(0.2 * (topY - bottomY + 1)),
                        rightX + (int)(0.2 * (rightX - leftX + 1)), topY + (int)(0.3 * (topY - bottomY + 1)));
                    //Photo.seamlessClone(warpMat, resultMat, maskMat,
                    //    new Point((ssdResults[i].left + ssdResults[i].right) / 2 * imageWidth, imageHeight - (ssdResults[i].top + ssdResults[i].bottom) / 2 * imageHeight),
                    //    resultMat, Photo.NORMAL_CLONE);
                }
            }
            stopWatch.Stop();
            RightText.text += "remove time = " + stopWatch.Elapsed.TotalMilliseconds + "   ";
            stopWatch.Reset();
        }
        catch(Exception ex)
        {
            RightText.text += "remove" + ex.Message;
        }
        //DoRightImage(imageTexture);
    }

    /// <summary>
    // Get Images
    /// </summary>
    public void ShotImage()
    {
        string shotName = DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss-ff");
        Debug.Log("shot button clicked!");
        //StartCoroutine(GetImage());
        Texture2D t = imageView.texture as Texture2D;
        //t = RotateN90(t);
        byte[] byt = t.EncodeToPNG();
        File.WriteAllBytes(Application.persistentDataPath + "/input" + shotName + ".png", byt);
        //File.WriteAllBytes(Application.persistentDataPath + "/photo" + Time.time + ".png", byt);
        t = resultView.texture as Texture2D;
        //t = RotateN90(t);
        byt = t.EncodeToPNG();
        File.WriteAllBytes(Application.persistentDataPath + "/result" + shotName + ".png", byt);

        //Texture2D image = new Texture2D(webCam.width, webCam.height, TextureFormat.RGBA32, false);
        //image.SetPixels(webCam.GetPixels());
        //image.Apply();
        //imageView.texture = image;
        //Mat input = new Mat(1080, 1080, CvType.CV_8UC4);
        //Utils.texture2DToMat(image, input);

        //deepLab.Invoke(webCam);
        //Texture2D mask0 = deepLab.GetResultTexture2D() as Texture2D;
        //Mat mask1 = new Mat(mask0.height, mask0.width, CvType.CV_8UC4);
        //Utils.texture2DToMat(mask0, mask1);
        //Mat mask = new Mat(1080, 1080, CvType.CV_8UC4);
        //Imgproc.resize(mask1, mask, srcSize);
        //Imgcodecs.imwrite(Application.streamingAssetsPath + "/inputNew" + shotName + ".png", input);
        //Imgcodecs.imwrite(Application.streamingAssetsPath + "/maskNew" + shotName + ".png", mask);
        //if(AppState > 0) {
        //    AppState = 0;
        //    webCam.Play();
        //    InfoText.text = "Capturing";
        //    //ShotButton.GetComponent<Text>().text = "重拍";
        //}
        //else {
        //    AppState = 1;
        //    webCam.Pause();
        //    InfoText.text = "Captured";
        //    StartCoroutine(GetImage());
        //    //ShotButton.GetComponent<Text>().text = "拍照";
        //}
    }

    // linear image fusion 
    public void linearClone(int leftX, int bottomY, int rightX, int topY)
    {
        leftX = leftX < 0 ? 0 : leftX;
        rightX = rightX > imageWidth - 1 ? imageWidth - 1 : rightX;
        bottomY = bottomY < 0 ? 0 : bottomY;
        topY = topY > imageHeight - 1 ? imageHeight - 1 : topY;
        stopWatch.Start();
        //Texture2D subTexture = new Texture2D(rightX - leftX + 1, topY - bottomY + 1, TextureFormat.RGB24, false);

        try
        {
            float[] leftDif = new float[3];
            float[] rightDif = new float[3];
            float[] topDif = new float[3];
            float[] bottomDif = new float[3];
            for (int i = bottomY; i <= topY; i++)
            {
                leftDif[0] += warpTexture.GetPixel(leftX, i).r - imageTexture.GetPixel(leftX, i).r;
                leftDif[1] += warpTexture.GetPixel(leftX, i).g - imageTexture.GetPixel(leftX, i).g;
                leftDif[2] += warpTexture.GetPixel(leftX, i).b - imageTexture.GetPixel(leftX, i).b;
                rightDif[0] += warpTexture.GetPixel(rightX, i).r - imageTexture.GetPixel(rightX, i).r;
                rightDif[1] += warpTexture.GetPixel(rightX, i).g - imageTexture.GetPixel(rightX, i).g;
                rightDif[2] += warpTexture.GetPixel(rightX, i).b - imageTexture.GetPixel(rightX, i).b;
            }
            for (int j = leftX; j <= rightX; j++)
            {
                topDif[0] += warpTexture.GetPixel(j, topY).r - imageTexture.GetPixel(j, topY).r;
                topDif[1] += warpTexture.GetPixel(j, topY).g - imageTexture.GetPixel(j, topY).g;
                topDif[2] += warpTexture.GetPixel(j, topY).b - imageTexture.GetPixel(j, topY).b;
                bottomDif[0] += warpTexture.GetPixel(j, bottomY).r - imageTexture.GetPixel(j, bottomY).r;
                bottomDif[1] += warpTexture.GetPixel(j, bottomY).g - imageTexture.GetPixel(j, bottomY).g;
                bottomDif[2] += warpTexture.GetPixel(j, bottomY).b - imageTexture.GetPixel(j, bottomY).b;
            }
            leftDif[0] /= (float)(topY - bottomY + 1);
            leftDif[1] /= (float)(topY - bottomY + 1);
            leftDif[2] /= (float)(topY - bottomY + 1);
            rightDif[0] /= (float)(topY - bottomY + 1);
            rightDif[1] /= (float)(topY - bottomY + 1);
            rightDif[2] /= (float)(topY - bottomY + 1);
            topDif[0] /= (float)(rightX - leftX + 1);
            topDif[1] /= (float)(rightX - leftX + 1);
            topDif[2] /= (float)(rightX - leftX + 1);
            bottomDif[0] /= (float)(rightX - leftX + 1);
            bottomDif[1] /= (float)(rightX - leftX + 1);
            bottomDif[2] /= (float)(rightX - leftX + 1);
            LeftText.text += leftDif.ToString() + " ";
            LeftText.text += topDif.ToString() + " ";
            LeftText.text += rightDif.ToString() + " ";
            LeftText.text += bottomDif.ToString() + " ";
            Color warpColor = new Color();
            Color imageColor = new Color();
            for (int i = bottomY;i<=topY;i++) {
                for (int j = leftX;j<=rightX;j++) {
                    imageColor = imageTexture.GetPixel(j, i);
                    warpColor = warpTexture.GetPixel(j, i);
                    Color newColor = new Color();
                    newColor.a = 1;
                    newColor.r = warpColor.r - leftDif[0] * (rightX - j) / (rightX - leftX + topY - bottomY)
                        - rightDif[0] * (j - leftX) / (rightX - leftX + topY - bottomY)
                        - bottomDif[0] * (topY - i) / (rightX - leftX + topY - bottomY)
                        - topDif[0] * (i - bottomY) / (rightX - leftX + topY - bottomY);
                    newColor.g = warpColor.g - leftDif[1] * (rightX - j) / (rightX - leftX + topY - bottomY)
                        - rightDif[1] * (j - leftX) / (rightX - leftX + topY - bottomY)
                        - bottomDif[1] * (topY - i) / (rightX - leftX + topY - bottomY)
                        - topDif[1] * (i - bottomY) / (rightX - leftX + topY - bottomY);
                    newColor.b = warpColor.b - leftDif[2] * (rightX - j) / (rightX - leftX + topY - bottomY)
                        - rightDif[2] * (j - leftX) / (rightX - leftX + topY - bottomY)
                        - bottomDif[2] * (topY - i) / (rightX - leftX + topY - bottomY)
                        - topDif[2] * (i - bottomY) / (rightX - leftX + topY - bottomY);
                    //newColor.g = warpColor.g - leftDif[i - bottomY, 1] * (rightX - j) / (rightX - leftX);
                    //newColor.b = warpColor.b - leftDif[i - bottomY, 2] * (rightX - j) / (rightX - leftX);
                    if (leftX > 30 && j < leftX + 30 && j - leftX <= i - bottomY && j - leftX <= topY - i)
                    {
                        newColor.r = imageColor.r * (30 - j + leftX) / 30f + newColor.r * (j - leftX) / 30f;
                        newColor.g = imageColor.g * (30 - j + leftX) / 30f + newColor.g * (j - leftX) / 30f;
                        newColor.b = imageColor.b * (30 - j + leftX) / 30f + newColor.b * (j - leftX) / 30f;
                    }
                    else if (rightX < imageWidth - 30 && j > rightX - 30 && rightX - j <= i - bottomY && rightX - j <= topY - i)
                    {
                        newColor.r = imageColor.r * (30f + j - rightX) / 30f + newColor.r * (rightX - j) / 30f;
                        newColor.g = imageColor.g * (30f + j - rightX) / 30f + newColor.g * (rightX - j) / 30f;
                        newColor.b = imageColor.b * (30f + j - rightX) / 30f + newColor.b * (rightX - j) / 30f;
                    }
                    else if (topY < imageHeight - 30 && i > topY - 30 && topY - i <= j - leftX && topY - i <= rightX - j)
                    {
                        newColor.r = imageColor.r * (30f + i - topY) / 30f + newColor.r * (topY - i) / 30f;
                        newColor.g = imageColor.g * (30f + i - topY) / 30f + newColor.g * (topY - i) / 30f;
                        newColor.b = imageColor.b * (30f + i - topY) / 30f + newColor.b * (topY - i) / 30f;
                    }
                    else if (bottomY > 30 &&i < bottomY + 30 && i - bottomY <= j - leftX && i - bottomY <= rightX - j)
                    {
                        newColor.r = imageColor.r * (30f - i + bottomY) / 30f + newColor.r * (i - bottomY) / 30f;
                        newColor.g = imageColor.g * (30f - i + bottomY) / 30f + newColor.g * (i - bottomY) / 30f;
                        newColor.b = imageColor.b * (30f - i + bottomY) / 30f + newColor.b * (i - bottomY) / 30f;
                    }
                    //else
                    //{
                    //    newColor.r = warpColor.r;
                    //    newColor.g = warpColor.g;
                    //    newColor.b = warpColor.b;
                    //}
                    //newColor.r = warpColor.r;
                    //newColor.g = warpColor.g;
                    //newColor.b = warpColor.b;
                    resultTexture.SetPixel(j, i, newColor);


                    //warpTexture.GetPixel(i, j);
                }
            }
            resultTexture.Apply();

            //int[] warpData = new int[warpMat.rows() * warpMat.cols() * warpMat.channels()];
            //warpMat.get(0, 0, warpData);
            //RightText.text += warpData[0] + " " + warpData[1] + warpData[2] + " ";
        }
        catch(Exception ex)
        {
            RightText.text += "scan " + ex.Message;
        }

        //warpMat.copyTo(resultMat, maskMat);
        //int[] inputData = new int[3];

        //for (int i = bottomY;i<=topY;i++)
        //{
        //    for (int j = leftX;j<=rightX;j++)
        //    {
        //        warpMat.get(i, j, warpData);
        //        inputMat.get(i, j, inputData);
        //    }
        //}

        stopWatch.Stop();
        RightText.text += "scan time = " + stopWatch.Elapsed.TotalMilliseconds;
        stopWatch.Reset();
    }
    // get lib images(locate, search and detect)
    // locating ad searching are removed now to make demo videos separatly
    IEnumerator LibProcess()
    {
        yield return new WaitForEndOfFrame();
        try
        {
            imageTexture = new Texture2D(webCam.width, webCam.height, TextureFormat.RGBA32, false);
            imageTexture.SetPixels(webCam.GetPixels());
            imageTexture.Apply();
            imageView.texture = imageTexture;
            //Utils.texture2DToMat(imageTexture, inputMat);

            libMat = Imgcodecs.imread(Application.persistentDataPath + "/lib.jpg");
            //Scalar diffMean = Core.mean(inputMat) - Core.mean(libMat);
            //Scalar diffMean = new Scalar(diffMean0.val[2], diffMean0.val[1], diffMean0.val[0]);
            //Scalar diffMean = new Scalar(50, 50, 50);
            //Core.add(libMat, diffMean, libMat);
            //Imgproc.cvtColor(libMat, libMat, Imgproc.COLOR_RGB2BGR);
            //LeftText.text += diffMean.ToString();
            //LeftText.text = "ilu done";
            //LeftText.text += Core.mean(libMat).ToString();
            //LeftText.text += Core.mean(inputMat).ToString();
            //LeftText.text += (Core.mean(inputMat) - Core.mean(libMat)).ToString();
            //Imgcodecs.imwrite(Application.persistentDataPath + "/libNew.png", libMat);
            detector.detectAndCompute(libMat, emptyMask, libKeyPointsMat, libDescriptor);
            libKeyPointsArray = libKeyPointsMat.toArray();
            LeftText.text += libKeyPointsArray.Length.ToString();
            //Texture2D newTexture = new Texture2D(libMat.cols(), libMat.rows(), TextureFormat.RGBA32, false);
            //Utils.matToTexture2D(libMat, newTexture);
            //resultView.texture = newTexture;
        }
        catch (Exception ex)
        {
            LeftText.text += ex.Message;
        }
        //t.ReadPixels( new Rect(0, 0, 899, 1199),0,0,false);
        //Texture2D t = new Texture2D(webCam.width, webCam.height, TextureFormat.RGBA32, false);
        //t.SetPixels(webCam.GetPixels());
        //t.Apply();
        //RawImageView.canvasRenderer.SetTexture(t);
        //Texture2D t = imageView.texture as Texture2D;
        //t = RotateN90(t);
        //byte[] byt = t.EncodeToPNG();
        //File.WriteAllBytes(Application.persistentDataPath + "/input" + DateTime.Now.ToShortTimeString() + ".png", byt);
        //File.WriteAllBytes(Application.persistentDataPath + "/photo" + Time.time + ".png", byt);
        //t = resultView.texture as Texture2D;
        //t = RotateN90(t);
        //byt = t.EncodeToPNG();
        //File.WriteAllBytes(Application.persistentDataPath + "/mask" + DateTime.Now.ToShortTimeString() + ".png", byt);
        //Mat imgMat = new Mat(t.height, t.width, CvType.CV_8UC4);
        //Utils.texture2DToMat(t, imgMat);
        //RightText.text = "image" + imgMat.size().ToString();
    }

    //public unsafe void ShotImage()
    //{
    //    Debug.Log("shoting image!");
    //    XRCameraImage image;
    //    if (!cameraManager.TryGetLatestImage(out image))
    //    {
    //        return;
    //    }
    //    imageInfo.text = string.Format(
    //        "Image info:\n\twidth: {0}\n\theight: {1}\n\tplaneCount: {2}\n\ttimestamp: {3}\n\tformat: {4}",
    //        image.width, image.height, image.planeCount, image.timestamp, image.format);
    //    var format = TextureFormat.RGBA32;
    //    texture = new Texture2D(image.width, image.height, format, false);
    //    var conversionParams = new XRCameraImageConversionParams(image, format, CameraImageTransformation.None);
    //    var rawTextureData = texture.GetRawTextureData<byte>();
    //    try
    //    {
    //        image.Convert(conversionParams, new IntPtr(rawTextureData.GetUnsafePtr()), rawTextureData.Length);
    //    }
    //    finally
    //    {
    //        // We must dispose of the XRCameraImage after we're finished
    //        // with it to avoid leaking native resources.
    //        image.Dispose();
    //    }

    //    // Texture2D allows us write directly to the raw texture data
    //    // This allows us to do the conversion in-place without making any copies.
    //    texture.Apply();
    //    Texture2D newTexture = RotateTexture(texture);
    //    ImageView.texture = newTexture;
    //    byte[] byt = newTexture.EncodeToPNG();
    //    File.WriteAllBytes(Application.persistentDataPath + "/input.png", byt);
    //    //File.WriteAllBytes(Application.persistentDataPath + "/result" + Time.time.ToString().Split('.')[0] + "_" + Time.time.ToString().Split('.')[1] + ".png", byt);
    //}

    void OnDestroy() {
        deepLab?.Dispose();
        ssd?.Dispose();
    }
    // Used for Texture rotating
    public Texture2D RotateP90(Texture2D texture)
    {
        var format = TextureFormat.RGBA32;
        Texture2D newTexture = new Texture2D(texture.height, texture.width, format, false);
        for (int i = 0;i<texture.height;i++) {
            for (int j = 0;j<texture.width;j++) {
                newTexture.SetPixel(i, j, texture.GetPixel(j, texture.height - 1 - i));
            }
        }
        newTexture.Apply();
        return newTexture;
    }
    public Texture2D RotateN90(Texture2D texture)
    {
        var format = TextureFormat.RGBA32;
        Texture2D newTexture = new Texture2D(texture.height, texture.width, format, false);
        for (int i = 0;i<texture.height;i++) {
            for (int j = 0;j<texture.width;j++) {
                newTexture.SetPixel(i, j, texture.GetPixel(texture.width - 1 - j, i));
            }
        }
        newTexture.Apply();
        return newTexture;
    }
    // Update is called once per frame
    //IEnumerator CallCamera()
    //{
    //    yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
    //    if (Application.HasUserAuthorization(UserAuthorization.WebCam)) {
    //        if (webCam != null) {
    //            webCam.Stop();
    //        }

    //        WebCamDevice[] camDevices = WebCamTexture.devices;
    //        string deviceName = camDevices[0].name;
    //        webCam = new WebCamTexture(deviceName, 1080, 1080, 30);
    //        webCam.Play();
    //        int angle = webCam.videoRotationAngle;
    //        imageView.transform.Rotate(new Vector3(0, 0, -angle));
    //    }
    //}    
}
