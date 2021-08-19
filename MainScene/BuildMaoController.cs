using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using easyar;
using System;

public class BuildMaoController : MonoBehaviour
{
    private ARSession session;
    private SparseSpatialMapWorkerFrameFilter mapWorker;
    private SparseSpatialMapController map;
    [SerializeField]
    private Text infoText;
    [SerializeField]
    private Button SaveButton;
    [SerializeField]
    private Button ShowButton;
    [SerializeField]
    private Button RedoButton;
    // Start is called before the first frame update
    void Start()
    {
        session = FindObjectOfType<ARSession>();
        mapWorker = FindObjectOfType<SparseSpatialMapWorkerFrameFilter>();
        map = FindObjectOfType<SparseSpatialMapController>();
        //SaveButton = GameObject.Find("/Canvas/SaveButton").GetComponent<Button>();
        //ShowButton = GameObject.Find("/Canvas/ShowButton").GetComponent<Button>();
        //RedoButton = GameObject.Find("/Canvas/RedoButton").GetComponent<Button>();
        //infoText = GameObject.Find("/Canvas/InfoTest").GetComponent<Text>();

        SaveButton.onClick.AddListener(SaveMap);
        SaveButton.interactable = false;

        session.WorldRootController.TrackingStatusChanged += OnTrackingStatusChanged;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    /// <summary>
    /// 保存地图方法
    /// </summary>
    private void SaveMap()
    {
        SaveButton.interactable = false;
        //注册地图保存结果反馈事件
        mapWorker.BuilderMapController.MapHost += SaveMapHostBack;
        //保存地图
        try
        {
            //保存地图
            mapWorker.BuilderMapController.Host("MyMap" + DateTime.Now.ToString("yyyyMMddHHmm"), null);
            infoText.text = "开始保存地图，请稍等。";
        }
        catch (Exception ex)
        {
            SaveButton.interactable = true;
            infoText.text = "保存出错：" + ex.Message;
        }
    }
    /// <summary>
    /// 保存地图反馈
    /// </summary>
    /// <param name="mapInfo">地图信息</param>
    /// <param name="isSuccess">成功标识</param>
    /// <param name="error">错误信息</param>
    private void SaveMapHostBack(SparseSpatialMapController.SparseSpatialMapInfo mapInfo, bool isSuccess, string error)
    {
        Debug.Log("save button clicked");
        if (isSuccess) {
            PlayerPrefs.SetString("MapID", mapInfo.ID);
            PlayerPrefs.SetString("MapName", mapInfo.Name);
            infoText.text = "地图保存成功。\r\nMapID：" + mapInfo.ID + "\r\nMapName：" + mapInfo.Name;
        }
        else {
            SaveButton.interactable = true;
            infoText.text = "地图保存出错：" + error;
        }
    }
    private void OnTrackingStatusChanged(MotionTrackingStatus status)
    {
        if(status == MotionTrackingStatus.Tracking) {
            infoText.text = "进入跟踪状态";
            SaveButton.interactable = true;
        }
        else {
            infoText.text = "退出跟踪状态";
            SaveButton.interactable = false;
        }
    }
}
