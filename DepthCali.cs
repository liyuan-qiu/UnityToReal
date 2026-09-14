using System;
using System.IO;
using System.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

public class SimplePhotoCapture : MonoBehaviour
{
    [Header("材质设置")]
    public Material depthMaterial;
    
    private Camera mainCamera;
    private HDAdditionalCameraData hdData;
    private RenderTexture sourceRT;
    private string saveDirectory;

    void Start()
    {
        mainCamera = GetComponent<Camera>();
        hdData = GetComponent<HDAdditionalCameraData>();

        // 确保 HDRP 相机数据组件存在
        if (hdData == null)
        {
            hdData = gameObject.AddComponent<HDAdditionalCameraData>();
        }

        // 开启必要的 Frame Settings
        var frameSettings = hdData.renderingPathCustomFrameSettings;
        frameSettings.SetEnabled(FrameSettingsField.OpaqueObjects, true);
        frameSettings.SetEnabled(FrameSettingsField.TransparentObjects, true);
        frameSettings.SetEnabled(FrameSettingsField.ShadowMaps, true);
        frameSettings.SetEnabled(FrameSettingsField.CustomPass, true);

        // 开启深度图模式
        mainCamera.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.DepthNormals;
        mainCamera.clearFlags = CameraClearFlags.Depth;

        // 初始化深度图用的 RenderTexture
        sourceRT = new RenderTexture(mainCamera.pixelWidth, mainCamera.pixelHeight, 24, RenderTextureFormat.ARGBFloat);
        sourceRT.Create();

        // 将相机的裁剪面数据传递给深度材质
        if (depthMaterial != null)
        {
            depthMaterial.SetFloat("_NearPlane", mainCamera.nearClipPlane);
            depthMaterial.SetFloat("_FarPlane", mainCamera.farClipPlane);
        }

        // 设置保存路径为项目根目录下的 Screenshots 文件夹
        saveDirectory = Path.Combine(Application.dataPath, "../Screenshots");
        Directory.CreateDirectory(saveDirectory);
    }

    void Update()
    {
        // 按下 P 键触发拍照
        if (Input.GetKeyDown(KeyCode.P))
        {
            StartCoroutine(CaptureSinglePhoto());
        }
    }

    private IEnumerator CaptureSinglePhoto()
    {
        // 等待当前帧渲染完毕，确保能抓取到完整的画面和深度信息
        yield return new WaitForEndOfFrame();

        // 使用时间戳命名，防止覆盖
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        
        CaptureRGB(timestamp);
        CaptureDepth(timestamp);
    }

    private void CaptureRGB(string timestamp)
    {
        RenderTexture rt = new RenderTexture(mainCamera.pixelWidth, mainCamera.pixelHeight, 24);
        RenderTexture previousRT = mainCamera.targetTexture;
        mainCamera.targetTexture = rt;
        
        mainCamera.Render();
        
        Texture2D screenshot = new Texture2D(mainCamera.pixelWidth, mainCamera.pixelHeight, TextureFormat.RGB24, false);
        RenderTexture.active = rt;
        screenshot.ReadPixels(new Rect(0, 0, mainCamera.pixelWidth, mainCamera.pixelHeight), 0, 0);
        screenshot.Apply();

        // 恢复相机状态并清理
        mainCamera.targetTexture = previousRT;
        RenderTexture.active = null;

        byte[] bytes = screenshot.EncodeToPNG();
        string filename = Path.Combine(saveDirectory, $"RGB_{timestamp}.png");
        File.WriteAllBytes(filename, bytes);

        Destroy(screenshot);
        Destroy(rt);
        Debug.Log($"RGB 图片已保存至: {filename}");
    }

    private void CaptureDepth(string timestamp)
    {
        var depthTexture = Shader.GetGlobalTexture("_CameraDepthTexture");
        if (depthTexture == null)
        {
            Debug.LogError("当前帧无法获取到深度图 (_CameraDepthTexture)。");
            return;
        }

        // 使用你原本验证过的 RGBAFloat 格式，但补全了 active 激活逻辑以确保能读到像素
        Texture2D tex = new Texture2D(mainCamera.pixelWidth, mainCamera.pixelHeight, TextureFormat.RGBAFloat, false);
        
        Graphics.Blit(depthTexture, sourceRT, depthMaterial);
        
        // 必须激活 sourceRT 才能从中 ReadPixels
        RenderTexture.active = sourceRT;
        tex.ReadPixels(new Rect(0, 0, mainCamera.pixelWidth, mainCamera.pixelHeight), 0, 0);
        tex.Apply();
        
        RenderTexture.active = null;

        byte[] bytes = tex.EncodeToPNG();
        string filename = Path.Combine(saveDirectory, $"Depth_{timestamp}.png");
        File.WriteAllBytes(filename, bytes);

        Destroy(tex);
        Debug.Log($"Depth 图片已保存至: {filename}");
    }
}