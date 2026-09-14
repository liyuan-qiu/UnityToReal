using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;
using System.IO;

public class DepthRecorder : MonoBehaviour
{
    private Camera cam;
    private HDAdditionalCameraData hdData;
    private RenderTexture sourceRT;
    public Material depthMaterial;
    private int frameCount = 0;
    private bool isRecording = false;

    private void OnEnable()
    {
        cam = GetComponent<Camera>();

        hdData = GetComponent<HDAdditionalCameraData>();

        if (hdData == null)
        {
            hdData = gameObject.AddComponent<HDAdditionalCameraData>();
        }

        // Important!!! Essential HDRP camera settings
        var frameSettings = hdData.renderingPathCustomFrameSettings;
        frameSettings.SetEnabled(FrameSettingsField.OpaqueObjects, true);
        frameSettings.SetEnabled(FrameSettingsField.TransparentObjects, true);
        frameSettings.SetEnabled(FrameSettingsField.ShadowMaps, true);
        frameSettings.SetEnabled(FrameSettingsField.CustomPass, true);

        // Enable depth texture, in theory, HDRP can automatically open the DepthTextureMode
        cam.depthTextureMode |= DepthTextureMode.Depth;

        // Step1: Create render texture
        sourceRT = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 24, RenderTextureFormat.ARGBFloat);
        sourceRT.Create();

        // Start recording coroutine
        isRecording = true;
        StartCoroutine(RecordDepthEverySecond());
    }

    private IEnumerator RecordDepthEverySecond()
    {
        while (isRecording)
        {
            yield return new WaitForSeconds(1.0f); // Wait for one second
            yield return StartCoroutine(WaitForDepthTexture());
        }
    }

    private IEnumerator WaitForDepthTexture()
    {
        // Wait until the end of the frame
        yield return new WaitForEndOfFrame();

        // Access the depth texture
        var depthTexture = Shader.GetGlobalTexture("_CameraDepthTexture");
        if (depthTexture == null)
        {
            Debug.LogError("Depth texture is not available.");
            yield break;
        }

        Debug.Log($"Depth texture exists: {depthTexture != null}");
        // Step 2: Create CPU-accessible texture first
        Texture2D tex = new Texture2D(cam.pixelWidth, cam.pixelHeight, TextureFormat.RGBAFloat, false);
        // Step3: Use the depth texture (e.g., copy it to another texture and apply the shader into it)
        Graphics.Blit(depthTexture, sourceRT, depthMaterial);
        // Step 4: Read the processed depth from GPU to CPU
        tex.ReadPixels(new Rect(0, 0, cam.pixelWidth, cam.pixelHeight), 0, 0);
        tex.Apply();
        // Save to file
        byte[] bytes = tex.EncodeToPNG();
        string depthDirectory = "recording/depth";
        Directory.CreateDirectory(depthDirectory);
        string filename = Path.Combine(depthDirectory, $"depth_{frameCount:D4}.png");
        File.WriteAllBytes(filename, bytes);

        Debug.Log($"Saved depth texture to: {filename}");
        frameCount++;
    }

    private void OnDisable()
    {
        // Stop recording
        isRecording = false;

        // Unsubscribe from the event
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
    }

    private void OnEndCameraRendering(ScriptableRenderContext context, Camera camera)
    {
        if (camera != cam) return;

        // Access the depth texture
        var depthTexture = Shader.GetGlobalTexture("_CameraDepthTexture");
        if (depthTexture == null)
        {
            Debug.LogError("Depth texture is not available. Check the following:");
            Debug.LogError("- Is the camera using HDRP?");
            Debug.LogError("- Is the depth texture enabled in the camera settings?");
            Debug.LogError("- Is the HD Additional Camera Data component properly configured?");
            return;
        }

        Debug.Log($"Depth texture exists: {depthTexture != null}");

        // Use the depth texture (e.g., copy it to another texture)
        Graphics.Blit(depthTexture, sourceRT);
    }
}
