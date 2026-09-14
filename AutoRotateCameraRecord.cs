using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

public class AutoRotateCameraRecord : MonoBehaviour
{
    public float speed = 1.0f;
    public float rotationSpeed = 0.2f;
    private StreamWriter sw;
    private String fileName;
    private float startTime;
    private bool isRecording = false;

    public float screenshotInterval = 2.0f;
    private float nextScreenshotTime = 0.0f;
    private string recordingDirectory;
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    public Material depthMaterial;
    private Camera mainCamera;
    public int frameCount = 0;
    private HDAdditionalCameraData hdData;
    private RenderTexture sourceRT;
    private Light sceneLight;
    public CameraBody cameraBody = new CameraBody();
    
    private float initialZ = -3.0f;
    private float currentZ = -3.0f;
    private float targetZ = -2.1f;
    private float zStep = 0.3f;
    private float fixedX = 0.84f;
    private float fixedY = 9.1f;
    private float initialTiltAngle = 160f;
    private float finalTiltAngle = 20f;

    
    private float rotationAngle = 0f; // Current rotation angle
    private float rotationStep = 10f;  // Rotate 2 degrees per step
    private float tiltStep = 10f;
    private float circleRadius = 3f;  // Distance from center in XZ plane
    private float cameraHeight = 1.5f;  // Y position
    private bool isAutoRotating = true;    
    void Start()
    {
        mainCamera = GetComponent<Camera>();
        hdData = GetComponent<HDAdditionalCameraData>();
        sceneLight = FindObjectOfType<Light>();

        if (hdData == null)
        {
            hdData = gameObject.AddComponent<HDAdditionalCameraData>();
        }

        var frameSettings = hdData.renderingPathCustomFrameSettings;
        frameSettings.SetEnabled(FrameSettingsField.OpaqueObjects, true);
        frameSettings.SetEnabled(FrameSettingsField.TransparentObjects, true);
        frameSettings.SetEnabled(FrameSettingsField.ShadowMaps, true);
        frameSettings.SetEnabled(FrameSettingsField.CustomPass, true);

        mainCamera.depthTextureMode |= DepthTextureMode.Depth;

        sourceRT = new RenderTexture(mainCamera.pixelWidth, mainCamera.pixelHeight, 24, RenderTextureFormat.ARGBFloat);
        sourceRT.Create();

        startTime = Time.time;

        // Create a timestamped directory for recordings
        recordingDirectory = Path.Combine("Recordings", DateTime.Now.ToString("yyyy-MM-dd-HH-mm"));
        Directory.CreateDirectory(recordingDirectory);

        fileName = Path.Combine(recordingDirectory, "Camera_position_rotation.csv");
        sw = new StreamWriter(fileName);
        sw.WriteLine("tX,tY,tZ,rX,rY,rZ,rW,time(s)");
        sw.Flush();

        initialPosition = transform.position;
        initialRotation = transform.rotation;

        // Save camera and light parameters
        SaveParameters();
        SaveCameraParameters();

        mainCamera.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.DepthNormals;
        mainCamera.clearFlags = CameraClearFlags.Depth;
        Debug.Log($"Near: {mainCamera.nearClipPlane}, Far: {mainCamera.farClipPlane}");
        Debug.Log($"Depth mode: {mainCamera.depthTextureMode}");
        if (depthMaterial != null)
        {
            depthMaterial.SetFloat("_NearPlane", mainCamera.nearClipPlane);
            depthMaterial.SetFloat("_FarPlane", mainCamera.farClipPlane);
        }

    }

    void SaveParameters()
    {

        string parametersPath = Path.Combine(recordingDirectory, "parameters.txt");
        using (StreamWriter paramWriter = new StreamWriter(parametersPath))
        {
            paramWriter.WriteLine("Camera Parameters:");
            paramWriter.WriteLine($"Vertical Field of View: {mainCamera.fieldOfView}°");
            //paramWriter.WriteLine($"Focal Length: {focalLength:F2}mm");
            paramWriter.WriteLine($"Near Clipping Plane: {mainCamera.nearClipPlane}");
            paramWriter.WriteLine($"Far Clipping Plane: {mainCamera.farClipPlane}");
            paramWriter.WriteLine($"Sensor Size X: {cameraBody.sensorSize.x}mm");
            paramWriter.WriteLine($"Sensor Size X: {cameraBody.sensorSize.y}mm");
	    
            if (sceneLight != null)
            {
                paramWriter.WriteLine("\n Light Parameters:");
                paramWriter.WriteLine($"Intensity: {sceneLight.intensity}");
                paramWriter.WriteLine($"Radius: {sceneLight.shadowRadius}");
            }
        }
    }
    void SaveCameraParameters()
    {
        string camPath = Path.Combine(recordingDirectory, "cam.txt");
        float fovRadians = mainCamera.fieldOfView * Mathf.Deg2Rad; // Convert FOV from degrees to radians
        float focalLength = 480 / (2 * Mathf.Tan(fovRadians / 2)); // Calculate focal length in mm
        using (StreamWriter writer = new StreamWriter(camPath))
        {

            writer.WriteLine($"{focalLength:F2},0,320,0,{focalLength:F2},240,0,0,1");
        }
    }    
    /*
    void UpdateCameraPosition()
    {
        // Convert angle to radians
        float angleRad = rotationAngle * Mathf.Deg2Rad;

        // Calculate new position
        float x = circleRadius * Mathf.Cos(angleRad);
        float z = circleRadius * Mathf.Sin(angleRad);

        // Update camera position
        transform.position = new Vector3(x, cameraHeight, z);

        // Make camera look at cube center
        transform.LookAt(new Vector3(0, 0, 0));
    }
    
    IEnumerator AutoRotateCamera()
    {
        while (isAutoRotating)
        {
            // Update rotation angle
            rotationAngle += rotationStep;
            if (rotationAngle >= 360f) rotationAngle -= 360f;

            // Update camera position and rotation
            UpdateCameraPosition();

            // If recording is enabled, capture frames
            if (isRecording && Time.time >= nextScreenshotTime)
            {
                nextScreenshotTime = Time.time + screenshotInterval;
                CaptureFrames();
                RecordTransform();
            }

            // Wait for next frame
            yield return new WaitForSeconds(4.0f); // Adjust timing as needed
        }
    }
    */
    IEnumerator AutoRotateCamera()
    {
        Debug.Log("Starting Auto Rotation");
        isRecording = true;
        frameCount = 0;
    
        float tiltAngle = initialTiltAngle; // Fixed angle with world Z-axis
    
        while (isAutoRotating && currentZ < targetZ)
        {
            // Set fixed position for this Z level
            transform.position = new Vector3(fixedX, fixedY, currentZ);
            Debug.Log($"Current Z: {currentZ}, Position: {transform.position}");
    
            // Rotate around world Z-axis while maintaining tilt angle
            for (float rotationAngle = 0; rotationAngle <= 360f; rotationAngle += rotationStep)
            {
                if (!isAutoRotating) break;
    
                // Create rotation that first tilts by tiltAngle around X, then rotates around Z
                Quaternion tilt = Quaternion.Euler(tiltAngle, 0, 0);
                Quaternion rotation = Quaternion.Euler(0, 0, rotationAngle);
                transform.rotation = rotation * tilt;
    
                Debug.Log($"Rotation Angle: {rotationAngle}, Tilt: {tiltAngle}");
    
                yield return new WaitForSecondsRealtime(0.1f);
                CaptureFrames();
                RecordTransform();
    
                yield return new WaitForSeconds(screenshotInterval-0.1f);
            }
    
            currentZ += zStep;
        }
    
        // At final Z position, transition from initialTiltAngle to finalTiltAngle degrees
        transform.position = new Vector3(fixedX, fixedY, targetZ);
        
        // Transition tilt from initialTiltAngle to finalTiltAngle degrees
        for (float newTiltAngle = initialTiltAngle; newTiltAngle >= finalTiltAngle; newTiltAngle -= tiltStep)
        {
            if (!isAutoRotating) break;
    
            transform.rotation = Quaternion.Euler(newTiltAngle, 0, transform.rotation.eulerAngles.z);
            
            yield return new WaitForSecondsRealtime(0.1f);
            CaptureFrames();
            RecordTransform();
    
            yield return new WaitForSeconds(screenshotInterval-0.1f);
        }
    
        // Final rotation at 30-degree tilt
        tiltAngle = finalTiltAngle;
	currentZ = targetZ;
        while (isAutoRotating && currentZ >= initialZ)
        {
            // Set fixed position for this Z level
            transform.position = new Vector3(fixedX, fixedY, currentZ);
            Debug.Log($"Current Z: {currentZ}, Position: {transform.position}");
    
            // Rotate around world Z-axis while maintaining tilt angle
            for (float rotationAngle = 0; rotationAngle <= 360f; rotationAngle += rotationStep)
            {
                if (!isAutoRotating) break;
    
                // Create rotation that first tilts by tiltAngle around X, then rotates around Z
                Quaternion tilt = Quaternion.Euler(tiltAngle, 0, 0);
                Quaternion rotation = Quaternion.Euler(0, 0, rotationAngle);
                transform.rotation = rotation * tilt;
    
                Debug.Log($"Rotation Angle: {rotationAngle}, Tilt: {tiltAngle}");
    
                yield return new WaitForSecondsRealtime(0.1f);
                CaptureFrames();
                RecordTransform();
    
                yield return new WaitForSeconds(screenshotInterval-0.1f);
            }
    
            currentZ -= zStep;
        }    
        isAutoRotating = false;
        isRecording = false;
        Debug.Log("Auto rotation complete");
    }
     
     
    void Update()
    {
        // Add new key control for auto-rotation
        if (Input.GetKeyDown(KeyCode.J))
        {
            isAutoRotating = !isAutoRotating;
            if (isAutoRotating)
            {
		// Reset parameters
		currentZ = -3.0f;
		frameCount = 0;
		StartCoroutine(AutoRotateCamera());
            }
        }
        // Check if Ctrl is pressed to switch between movement and rotation
        bool isCtrlPressed = Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl);

        //float rotationSpeed = 0.2f; // Rotation speed
    
        if (isCtrlPressed)
        {
            // Rotation Mode (Yaw and Pitch only)
            if (Input.GetKey(KeyCode.LeftArrow))
            {
                // Yaw left (rotate around Y-axis)
                transform.Rotate(0, -rotationSpeed, 0, Space.World); // Global yaw
                // For local yaw, use Space.Self
            }
            if (Input.GetKey(KeyCode.RightArrow))
            {
                // Yaw right
                transform.Rotate(0, rotationSpeed, 0, Space.World);
            }
            if (Input.GetKey(KeyCode.UpArrow))
            {
                // Pitch up (rotate around X-axis)
                transform.Rotate(-rotationSpeed, 0, 0, Space.Self); // Local pitch
            }
            if (Input.GetKey(KeyCode.DownArrow))
            {
                // Pitch down
                transform.Rotate(rotationSpeed, 0, 0, Space.Self);
            }
        }
        else
        {
            // 6-Axis Movement Mode
            if (Input.GetKey(KeyCode.LeftArrow))
            {
                transform.position += Vector3.right * speed * Time.deltaTime; // Global left
                // For local left, use -transform.right
            }
            if (Input.GetKey(KeyCode.RightArrow))
            {
                transform.position += Vector3.left * speed * Time.deltaTime; // Global right
                // For local right, use transform.right
            }
            if (Input.GetKey(KeyCode.UpArrow))
            {
                transform.position += Vector3.back * speed * Time.deltaTime; // Global forward
                // For local forward, use transform.forward
            }
            if (Input.GetKey(KeyCode.DownArrow))
            {
                transform.position += Vector3.forward * speed * Time.deltaTime; // Global back
                // For local back, use -transform.forward
            }
            if (Input.GetKey(KeyCode.W))
            {
                transform.position += Vector3.up * speed * Time.deltaTime; // Global up
                // For local up, use transform.up
            }
            if (Input.GetKey(KeyCode.S))
            {
                transform.position += Vector3.down * speed * Time.deltaTime; // Global down
                // For local down, use -transform.up
            }
        }
    
        // Recording toggle (unchanged)
        if (Input.GetKeyDown(KeyCode.R))
        {
            isRecording = !isRecording;
            if (isRecording)
            {
                frameCount = 0;
                StartCoroutine(CaptureRoutine());
                Debug.Log("Recording started");
            }
            else
            {
                Debug.Log("Recording stopped");
            }
        }
    }
    
    IEnumerator CaptureRoutine()
    {
        while (isRecording)
        {
            yield return new WaitForEndOfFrame();

            if (Time.time >= nextScreenshotTime)
            {
                nextScreenshotTime = Time.time + screenshotInterval;
                CaptureFrames();
                RecordTransform();
            }
        }
    }

    void CaptureFrames()
    {
        SaveMainImage();
        SaveDepth();
    }

    void SaveMainImage()
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
        mainCamera.Render();
        RenderTexture.active = null;

        byte[] bytes = screenshot.EncodeToPNG();
        string filename = Path.Combine(recordingDirectory, $"Frames/photo_{frameCount:D4}.png");
        Directory.CreateDirectory(Path.Combine(recordingDirectory, "Frames"));
        File.WriteAllBytes(filename, bytes);
        Destroy(screenshot);
        Destroy(rt);
    }

    void SaveDepth()
    {
        var depthTexture = Shader.GetGlobalTexture("_CameraDepthTexture");
        if (depthTexture == null)
        {
            Debug.LogError("Depth texture is not available.");
            return;
        }

        Texture2D tex = new Texture2D(mainCamera.pixelWidth, mainCamera.pixelHeight, TextureFormat.RGBAFloat, false);
        Graphics.Blit(depthTexture, sourceRT, depthMaterial);
        tex.ReadPixels(new Rect(0, 0, mainCamera.pixelWidth, mainCamera.pixelHeight), 0, 0);
        tex.Apply();

        // Calculate min and max depth
        float minDepth = float.MaxValue;
        float maxDepth = float.MinValue;
        Color[] pixels = tex.GetPixels();
        foreach (Color pixel in pixels)
        {
            float depth = pixel.r;
            if (depth < minDepth) minDepth = depth;
            if (depth > maxDepth) maxDepth = depth;
        }

        /*
        // Save depth range to parameters.txt
        string parametersPath = Path.Combine(recordingDirectory, "parameters.txt");
        using (StreamWriter paramWriter = File.AppendText(parametersPath))
        {
            paramWriter.WriteLine($"\nDepth Range for Frame {frameCount}:");
            paramWriter.WriteLine($"Minimum Depth: {minDepth}");
            paramWriter.WriteLine($"Maximum Depth: {maxDepth}");
        }
        */

        byte[] bytes = tex.EncodeToPNG();
        string filename = Path.Combine(recordingDirectory, $"Depths/depth_{frameCount:D4}.png");
        Directory.CreateDirectory(Path.Combine(recordingDirectory, "Depths"));
        File.WriteAllBytes(filename, bytes);

        Debug.Log($"Saved depth texture to: {filename}");
        frameCount++;
    }

    void RecordTransform()
    {
        //Vector3 relativePosition = transform.position - initialPosition;
        //Quaternion relativeRotation = Quaternion.Inverse(initialRotation) * transform.rotation;
        Vector3 relativePosition = transform.position;
        Quaternion relativeRotation =  transform.rotation;

        Debug.Log("Time: " + (Time.time - startTime).ToString("F2") +
                  " | Position: (" + 
                  relativePosition.x.ToString("F2") + ", " +
                  relativePosition.y.ToString("F2") + ", " +
                  relativePosition.z.ToString("F2") + ")" +
                  " | Rotation: (" +
                  relativeRotation.x.ToString("F2") + ", " +
                  relativeRotation.y.ToString("F2") + ", " +
                  relativeRotation.z.ToString("F2") + ", " +
                  relativeRotation.w.ToString("F2") + ")");

        sw.WriteLine(
            relativePosition.x.ToString() + "," +
            relativePosition.y.ToString() + "," +
            relativePosition.z.ToString() + "," +
            relativeRotation.x.ToString() + "," +
            relativeRotation.y.ToString() + "," +
            relativeRotation.z.ToString() + "," +
            relativeRotation.w.ToString() + "," +
            (Time.time - startTime).ToString()
        );
        sw.Flush();
    }

    void OnApplicationQuit()
    {
        if (sw != null)
        {
            sw.Close();
        }
    }
}
