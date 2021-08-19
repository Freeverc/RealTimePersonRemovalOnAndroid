using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System;
using easyar;
using System.IO;
using System.Linq;

public class LocalizeMapController : MonoBehaviour
{
    private ARSession session;
    private SparseSpatialMapWorkerFrameFilter mapWorker;
    [SerializeField]
    private SparseSpatialMapController map1;
    [SerializeField]
    private SparseSpatialMapController map2;
    [SerializeField]
    private SparseSpatialMapController map3;
    [SerializeField]
    private SparseSpatialMapController map4;
    [SerializeField]
    private Text infoText;
    [SerializeField]
    private Button localizeButton;
    [SerializeField]
    private Button saveButton;
    [SerializeField]
    private Button captureButton;
    [SerializeField]
    private Button segmentButton;
    [SerializeField]
    private Button removeButton;
    [SerializeField]
    private Button searchButton;
    [SerializeField]
    private Toggle hideUI;
    [SerializeField]
    private Toggle hideObjects;
    [SerializeField]
    private Toggle hidePoints;

    //private VIOCameraDeviceUnion camera;
    [SerializeField]
    private RawImage mainImage;
    private Texture2D mainTexture;

    [SerializeField]
    private VIOCameraDeviceUnion vioCamera;
    [SerializeField]
    private CameraImageRenderer cameraRenderer;
    private Action<Camera, RenderTexture> targetTextureEventHandler;

    private string mapName;
    private StreamWriter sw;

    void Start()
    {
        if(!Directory.Exists(Application.persistentDataPath + "/Input")) {
            Directory.CreateDirectory(Application.persistentDataPath + "/Input");
        }
        Screen.SetResolution(2400, 1080, false);
        //Screen.SetResolution(1080, 2400, false);
        session = FindObjectOfType<ARSession>();
        mapWorker = FindObjectOfType<SparseSpatialMapWorkerFrameFilter>();
        //map1 = GameObject.Find("SparseSpatialMap1").GetComponent<SparseSpatialMapController>();
        //map2 = GameObject.Find("SparseSpatialMap2").GetComponent<SparseSpatialMapController>();
        //map3 = GameObject.Find("SparseSpatialMap3").GetComponent<SparseSpatialMapController>();

        //inputID = GameObject.Find("/Canvas/InputID").GetComponent<Input>();
        //inputName = GameObject.Find("/Canvas/InputName").GetComponent<Input>();

        // set maps
        map1.MapManagerSource.ID = "cb46f1bd-3135-4dd7-b30e-5e8702939de7";
        map1.MapManagerSource.Name = "Map-2020-08-26-16-09-30-44";
        map2.MapManagerSource.ID = "d23287e3-4eb8-4757-af09-9ba2455c9963";
        map2.MapManagerSource.Name = "Map-2020-08-06-13-22-10-46";
        map3.MapManagerSource.ID = "29922695-17cc-4445-9f86-83c885675c71";
        map3.MapManagerSource.Name = "Map-2020-08-06-12-29-06-75";
        map4.MapManagerSource.ID = "2e5b9353-9868-447b-9691-79a46929074a";
        map4.MapManagerSource.Name = "Map-2020-08-05-20-44-03-33";

        //map1.MapManagerSource.ID = "3223cf17-718f-418d-9bcf-18d92d809c77";
        //map1.MapManagerSource.Name = "Map-2020-08-26-14-28-52-43";
        //map2.MapManagerSource.ID = "970fcb9e-9db3-4d52-97e1-6fb4ebe32411";
        //map2.MapManagerSource.Name = "Map-2020-08-26-14-27-07-99";
        //map3.MapManagerSource.ID = "42fc95db-65f7-4388-b360-dc4720b10887";
        //map3.MapManagerSource.Name = "Map-2020-08-06-13-32-39-60";
        //map4.MapManagerSource.ID = "9d5a1dc5-5e17-4cef-be16-d5267157afbd";
        //map4.MapManagerSource.Name = "Map-2020-08-26-14-36-19-58";

        map1.MapLoad += MapLoadBack;
        map1.MapLocalized += LocalizeMap;
        map1.MapStopLocalize += StopLocalizeMap;
        map2.MapLoad += MapLoadBack;
        map2.MapLocalized += LocalizeMap;
        map2.MapStopLocalize += StopLocalizeMap;
        map3.MapLoad += MapLoadBack;
        map3.MapLocalized += LocalizeMap;
        map3.MapStopLocalize += StopLocalizeMap;
        map4.MapLoad += MapLoadBack;
        map4.MapLocalized += LocalizeMap;
        map4.MapStopLocalize += StopLocalizeMap;

        localizeButton.onClick.AddListener(StartLocalize);
        //saveButton.onClick.AddListener(StopLocalize);
        saveButton.onClick.AddListener(SaveImage);
        captureButton.onClick.AddListener(GetImage);

        hidePoints.onValueChanged.AddListener(HidePoints);
        hideUI.onValueChanged.AddListener(HideUI);
        hideObjects.onValueChanged.AddListener(HideObjects);

        captureButton.interactable = false;

        targetTextureEventHandler = (camera, texture) =>
        {
            if (texture)
            {
                //CubeRenderer.material.mainTexture = texture;

                RenderTexture.active = texture;
                mainTexture = new Texture2D(texture.width, texture.height, TextureFormat.RGBA32, false);
                mainTexture.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0);
                mainTexture.Apply();
                byte[] byt = mainTexture.EncodeToPNG();
                //File.WriteAllBytes(Application.persistentDataPath + "/photo" + Time.time + ".png", byt);
                File.WriteAllBytes(Application.persistentDataPath + "Input/input.png", byt);
            }
            else
            {
                //CubeRenderer.material.mainTexture = cubeTexture;
                //if (SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Metal)
                //{
                //    CubeRenderer.transform.localScale = new Vector3(-1, -1, 1);
                //}
                //else
                //{
                //    CubeRenderer.transform.localScale = new Vector3(1, 1, 1);
                //}
            }
        };
        StartLocalize();
    }
    private void StartLocalize()
    {
        //if (inputID.text.Length > 0)
        //{
        //    map.MapManagerSource.ID = inputID.text;
        //    map.MapManagerSource.Name = inputName.text;
        //}
        infoText.text = "开始本地化地图";
        try
        {
            mapWorker.Localizer.startLocalization();
        }
        catch(Exception ex)
        {
            infoText.text += ex.Message;
        }
    }
    private void StopLocalize()
    {
        infoText.text = "停止本地化地图";
        mapWorker.Localizer.stopLocalization();
    }
    private void SaveImage()
    {
        ;
    }
    private void MapLoadBack(
        SparseSpatialMapController.SparseSpatialMapInfo mapInfo,
        bool isSuccess,
        string error
        )
    {
        if(isSuccess)
        {
            infoText.text += "地图" + mapInfo.Name + "加载成功";
        }
        else
        {
            infoText.text += "地图加载失败" + error;
        }
    }
    // 
    private void LocalizeMap()
    {
        infoText.text = "地图定位成功" + DateTime.Now.ToShortTimeString();
        mapName = mapWorker.LocalizedMap.MapInfo.Name;
        captureButton.interactable = true;
        easyar.Buffer b = mapWorker.Localizer.getPointCloudBuffer();
        string s = b.data().ToString();
        sw = new StreamWriter(Application.persistentDataPath + "/Input/MapInfo" + ".txt", true);
        sw.WriteLine(mapWorker.LocalizedMap.MapInfo.Name);
        sw.WriteLine(mapWorker.LocalizedMap.MapInfo.ID);
        sw.WriteLine(s);
        sw.Close();
    }

    // stop localizing
    private void StopLocalizeMap()
    {
        infoText.text = "地图停止定位" + DateTime.Now.ToShortTimeString();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    // hide UI
    private void HideUI(bool on)
    {
        if(on) {
            GameObject.Find("/Canvas/Panel").GetComponent<CanvasGroup>().alpha = 0;
        }
        else {
            GameObject.Find("/Canvas/Panel").GetComponent<CanvasGroup>().alpha = 1;
        }
    }
    // hide objects
    private void HideObjects(bool on)
    {
        List<GameObject> objects = GameObject.FindGameObjectsWithTag("TheObject").ToList();
        if (on) {
            foreach(var obj in objects)
            {
                obj.GetComponent<MeshRenderer>().enabled = false;
            }
            //GameObject.Find("/Cube0").GetComponent<MeshRenderer>().enabled = false;
            //GameObject.Find("/WorldRoot/Cube1").GetComponent<MeshRenderer>().enabled = false;
            //GameObject.Find("/WorldRoot/Capsulex+").GetComponent<MeshRenderer>().enabled = false;
            //GameObject.Find("/WorldRoot/Capsuley+").GetComponent<MeshRenderer>().enabled = false;
            //GameObject.Find("/WorldRoot/Capsulez+").GetComponent<MeshRenderer>().enabled = false;
        }
        else {
            foreach(var obj in objects)
            {
                obj.GetComponent<MeshRenderer>().enabled = true;
            }
            //GameObject.Find("/Cube0").GetComponent<MeshRenderer>().enabled = true;
            //GameObject.Find("/WorldRoot/Cube1").GetComponent<MeshRenderer>().enabled = true;
            //GameObject.Find("/WorldRoot/Capsulex+").GetComponent<MeshRenderer>().enabled = true;
            //GameObject.Find("/WorldRoot/Capsuley+").GetComponent<MeshRenderer>().enabled = true;
            //GameObject.Find("/WorldRoot/Capsulez+").GetComponent<MeshRenderer>().enabled = true;
        }
    }
    // hide points
    private void HidePoints(bool on)
    {
        if(on) {
            mapWorker.LocalizedMap.ShowPointCloud = false;
        }
        else {
            mapWorker.LocalizedMap.ShowPointCloud = true;
        }
    }
    // get image
    private void GetImage()
    {
        try {
            string imageName = Application.persistentDataPath + 
                "/Input/InputImage-" + DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss-ff") + ".png";

            mainTexture = new Texture2D(Screen.width, Screen.height, TextureFormat.RGBA32, false);
            mainTexture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
            mainTexture.Apply();
            mainImage.texture = mainTexture;
            byte[] byt = mainTexture.EncodeToPNG();
            File.WriteAllBytes(imageName, byt);
            GameObject.Find("/Canvas/Panel").GetComponent<CanvasGroup>().alpha = 1;

            string pc = GameObject.Find("Main Camera").transform.position.ToString();
            string rc = GameObject.Find("Main Camera").transform.rotation.eulerAngles.ToString();
            string pm = mapWorker.LocalizedMap.transform.position.ToString();
            string rm = mapWorker.LocalizedMap.transform.rotation.eulerAngles.ToString();
            sw = new StreamWriter(Application.persistentDataPath + "/Input/imageView" + ".txt", true);
            sw.WriteLine(imageName);
            sw.WriteLine(mapName);
            sw.WriteLine(pc);
            sw.WriteLine(rc);
            sw.WriteLine(pm);
            sw.WriteLine(rm);
            sw.Close();
            infoText.text = "图像已保存";
        } catch(Exception ex)
        {
            infoText.text = "图像保存失败 " + ex.Message;
        }
    }
}
