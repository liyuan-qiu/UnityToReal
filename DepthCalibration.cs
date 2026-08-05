using System;
using System.IO;
using System.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

public class DepthCalibration : MonoBehaviour
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

        if (hdData == null)
        {
            hdData = gameObject.AddComponent<HDAdditionalCameraData>();
        }

        var frameSettings = hdData.renderingPathCustomFrameSettings;
        frameSettings.SetEnabled(FrameSettingsField.OpaqueObjects, true);
        frameSettings.SetEnabled(FrameSettingsField.TransparentObjects, true);
        frameSettings.SetEnabled(FrameSettingsField.ShadowMaps, true);
        frameSettings.SetEnabled(FrameSettingsField.CustomPass, true);

        mainCamera.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.DepthNormals;
        mainCamera.clearFlags = CameraClearFlags.Depth;

        sourceRT = new RenderTexture(mainCamera.pixelWidth, mainCamera.pixelHeight, 24, RenderTextureFormat.ARGBFloat);
        sourceRT.Create();

        if (depthMaterial != null)
        {
            depthMaterial.SetFloat("_NearPlane", mainCamera.nearClipPlane);
            depthMaterial.SetFloat("_FarPlane", mainCamera.farClipPlane);
        }

        // 使用相对关系计算出 recording 文件夹的干净物理路径
        string relativePath = Path.Combine(Application.dataPath, "../recording");
        saveDirectory = Path.GetFullPath(relativePath);
        Directory.CreateDirectory(saveDirectory);
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.P))
        {
            StartCoroutine(CaptureSinglePhoto());
        }
    }

    private IEnumerator CaptureSinglePhoto()
    {
        yield return new WaitForEndOfFrame();

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
        // 增加安全检查，防止未赋值材质导致程序崩溃
        if (depthMaterial == null)
        {
            Debug.LogError("Depth Material 未赋值！请在 Inspector 面板中将材质拖入 Depth Material 槽位。");
            return;
        }

        var depthTexture = Shader.GetGlobalTexture("_CameraDepthTexture");
        if (depthTexture == null)
        {
            Debug.LogError("当前帧无法获取到深度图 (_CameraDepthTexture)。");
            return;
        }

        Texture2D tex = new Texture2D(mainCamera.pixelWidth, mainCamera.pixelHeight, TextureFormat.RGBAFloat, false);
        
        Graphics.Blit(depthTexture, sourceRT, depthMaterial);
        
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