using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

/// <summary>
/// Put on the Camera. Reads camera_pose_unity_cam2tag_face.csv, applies
/// unity_pos + unity_quat (preferred) or unity_rot euler, captures RGB + Depth
/// every 1s into Output Directory (Inspector). If empty, uses recording/<folder>.
/// Also writes unity_camera_quat_export.csv with the live camera quaternion
/// for comparison against CSV unity_quat_*.
/// Names: CamCoordTest_1_Unity.jpg , CamCoordTest_1_Depth.png
/// Press Start Key (default P) to begin the batch.
/// </summary>
public class PoseCsvAutoCapture : MonoBehaviour
{
    [Header("CSV")]
    [Tooltip("Absolute path, or relative to project root (parent of Assets).")]
    public string csvRelativePath = "camera_pose_unity_cam2tag_face.csv";

    [Header("Capture")]
    public Material depthMaterial;
    public float intervalSeconds = 1f;
    public KeyCode startKey = KeyCode.P;
    public bool autoStartOnPlay = false;
    [Tooltip("If CSV has unity_quat_*, apply quaternion instead of euler.")]
    public bool preferQuaternion = true;

    [Header("Output")]
    [Tooltip("Where RGB, Depth, and pose CSVs are written. Absolute path, or relative to project root (parent of Assets). Leave empty to use recording/<Output Folder Name>.")]
    public string outputDirectory = "";

    [Tooltip("Used only when Output Directory is empty: subfolder under ../recording/. Default = CSV file name without extension.")]
    public string outputFolderName = "camera_pose_unity_cam2tag_face";

    private Camera mainCamera;
    private HDAdditionalCameraData hdData;
    private RenderTexture sourceRT;
    private string saveDirectory;
    private string resolvedCsvPath;
    private bool isRunning;

    [Serializable]
    private class PoseRow
    {
        public string imageFile;
        public Vector3 position;
        public Vector3 euler;
        public bool hasQuat;
        public Quaternion quat;
        public float csvQuatX, csvQuatY, csvQuatZ, csvQuatW;
    }

    void Start()
    {
        mainCamera = GetComponent<Camera>();
        if (mainCamera == null)
        {
            Debug.LogError("PoseCsvAutoCapture must be on a GameObject with a Camera.");
            enabled = false;
            return;
        }

        hdData = GetComponent<HDAdditionalCameraData>();
        if (hdData == null)
            hdData = gameObject.AddComponent<HDAdditionalCameraData>();

        var frameSettings = hdData.renderingPathCustomFrameSettings;
        frameSettings.SetEnabled(FrameSettingsField.OpaqueObjects, true);
        frameSettings.SetEnabled(FrameSettingsField.TransparentObjects, true);
        frameSettings.SetEnabled(FrameSettingsField.ShadowMaps, true);
        frameSettings.SetEnabled(FrameSettingsField.CustomPass, true);

        mainCamera.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.DepthNormals;

        sourceRT = new RenderTexture(mainCamera.pixelWidth, mainCamera.pixelHeight, 24, RenderTextureFormat.ARGBFloat);
        sourceRT.Create();

        if (depthMaterial != null)
        {
            depthMaterial.SetFloat("_NearPlane", mainCamera.nearClipPlane);
            depthMaterial.SetFloat("_FarPlane", mainCamera.farClipPlane);
        }

        resolvedCsvPath = ResolveCsvPath(csvRelativePath);
        saveDirectory = ResolveOutputDirectory();
        Directory.CreateDirectory(saveDirectory);

        Debug.Log($"[PoseCsvAutoCapture] CSV: {resolvedCsvPath}");
        Debug.Log($"[PoseCsvAutoCapture] Save: {saveDirectory}");
        Debug.Log($"[PoseCsvAutoCapture] Press {startKey} to start (or enable Auto Start On Play).");

        if (autoStartOnPlay)
            StartCoroutine(RunCaptureSequence());
    }

    void Update()
    {
        if (!isRunning && Input.GetKeyDown(startKey))
            StartCoroutine(RunCaptureSequence());
    }

    void OnDestroy()
    {
        if (sourceRT != null)
        {
            sourceRT.Release();
            Destroy(sourceRT);
        }
    }

    private IEnumerator RunCaptureSequence()
    {
        if (isRunning)
            yield break;
        isRunning = true;

        if (!File.Exists(resolvedCsvPath))
        {
            Debug.LogError($"CSV not found: {resolvedCsvPath}");
            isRunning = false;
            yield break;
        }

        List<PoseRow> rows;
        try
        {
            rows = LoadPoseRows(resolvedCsvPath);
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to parse CSV: {e.Message}");
            isRunning = false;
            yield break;
        }

        if (rows.Count == 0)
        {
            Debug.LogError("CSV has no data rows.");
            isRunning = false;
            yield break;
        }

        Debug.Log($"[PoseCsvAutoCapture] Capturing {rows.Count} poses, interval={intervalSeconds}s");
        TryCopyInputPoseCsv();

        var export = new StringBuilder();
        export.AppendLine(
            "image_file,csv_quat_x,csv_quat_y,csv_quat_z,csv_quat_w," +
            "unity_quat_x,unity_quat_y,unity_quat_z,unity_quat_w," +
            "quat_dot_abs,pos_x,pos_y,pos_z,euler_x,euler_y,euler_z,applied_via");

        for (int i = 0; i < rows.Count; i++)
        {
            PoseRow row = rows[i];
            transform.position = row.position;
            string appliedVia;
            if (preferQuaternion && row.hasQuat)
            {
                transform.rotation = row.quat;
                appliedVia = "quaternion";
            }
            else
            {
                transform.rotation = Quaternion.Euler(row.euler);
                appliedVia = "euler";
            }

            // Let the camera / HDRP settle at the new pose.
            yield return null;
            yield return new WaitForEndOfFrame();

            Quaternion qLive = transform.rotation;
            // Match CSV convention: prefer w >= 0
            if (qLive.w < 0f)
                qLive = new Quaternion(-qLive.x, -qLive.y, -qLive.z, -qLive.w);

            float dotAbs = 0f;
            if (row.hasQuat)
            {
                float d = row.csvQuatX * qLive.x + row.csvQuatY * qLive.y
                          + row.csvQuatZ * qLive.z + row.csvQuatW * qLive.w;
                dotAbs = Mathf.Abs(d);
            }

            string stem = Path.GetFileNameWithoutExtension(row.imageFile); // CamCoordTest_1
            string unityName = $"{stem}_Unity";
            string depthName = $"{stem}_Depth";

            CaptureRGB(unityName);
            CaptureDepth(depthName);

            Vector3 e = transform.eulerAngles;
            export.AppendLine(string.Format(
                CultureInfo.InvariantCulture,
                "{0},{1:F9},{2:F9},{3:F9},{4:F9},{5:F9},{6:F9},{7:F9},{8:F9},{9:F6},{10:F9},{11:F9},{12:F9},{13:F6},{14:F6},{15:F6},{16}",
                row.imageFile,
                row.csvQuatX, row.csvQuatY, row.csvQuatZ, row.csvQuatW,
                qLive.x, qLive.y, qLive.z, qLive.w,
                dotAbs,
                transform.position.x, transform.position.y, transform.position.z,
                e.x, e.y, e.z,
                appliedVia));

            Debug.Log(
                $"[{i + 1}/{rows.Count}] {stem} via={appliedVia}  " +
                $"quatLive=({qLive.x:F4},{qLive.y:F4},{qLive.z:F4},{qLive.w:F4})  " +
                $"quatDotAbs={dotAbs:F4}");

            if (i < rows.Count - 1)
                yield return new WaitForSeconds(intervalSeconds);
        }

        string exportPath = Path.Combine(saveDirectory, "unity_camera_quat_export.csv");
        File.WriteAllText(exportPath, export.ToString(), Encoding.UTF8);
        TryCopyInputPoseCsv();
        Debug.Log($"[PoseCsvAutoCapture] Quaternion export: {exportPath}");
        Debug.Log($"[PoseCsvAutoCapture] Done. Files in: {saveDirectory}");
        isRunning = false;
    }

    private string ResolveOutputDirectory()
    {
        if (!string.IsNullOrWhiteSpace(outputDirectory))
        {
            string raw = outputDirectory.Trim().Trim('"');
            if (Path.IsPathRooted(raw))
                return Path.GetFullPath(raw);

            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            return Path.GetFullPath(Path.Combine(projectRoot, raw));
        }

        string folder = string.IsNullOrEmpty(outputFolderName)
            ? Path.GetFileNameWithoutExtension(resolvedCsvPath)
            : outputFolderName;
        return Path.GetFullPath(Path.Combine(Application.dataPath, "../recording", folder));
    }

    private void TryCopyInputPoseCsv()
    {
        if (string.IsNullOrEmpty(resolvedCsvPath) || !File.Exists(resolvedCsvPath))
            return;
        try
        {
            string dest = Path.Combine(saveDirectory, Path.GetFileName(resolvedCsvPath));
            if (!string.Equals(Path.GetFullPath(resolvedCsvPath), Path.GetFullPath(dest), StringComparison.OrdinalIgnoreCase))
                File.Copy(resolvedCsvPath, dest, overwrite: true);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[PoseCsvAutoCapture] Could not copy input pose CSV: {e.Message}");
        }
    }

    private static string ResolveCsvPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return path;

        if (Path.IsPathRooted(path) && File.Exists(path))
            return Path.GetFullPath(path);

        // Project root = parent of Assets
        string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        string candidate = Path.GetFullPath(Path.Combine(projectRoot, path));
        if (File.Exists(candidate))
            return candidate;

        // Also try next to Assets / StreamingAssets
        candidate = Path.GetFullPath(Path.Combine(Application.streamingAssetsPath, path));
        if (File.Exists(candidate))
            return candidate;

        candidate = Path.GetFullPath(Path.Combine(Application.dataPath, path));
        return candidate;
    }

    private static List<PoseRow> LoadPoseRows(string csvPath)
    {
        var rows = new List<PoseRow>();
        // utf-8-sig equivalent: skip BOM so first header is "image_file" not "\uFEFFimage_file"
        string text = File.ReadAllText(csvPath, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        if (text.Length > 0 && text[0] == '\uFEFF')
            text = text.Substring(1);

        string[] lines = text.Split(new[] { "\r\n", "\n", "\r" }, StringSplitOptions.None);
        if (lines.Length < 2)
            return rows;

        string headerLine = lines[0].Trim().TrimStart('\uFEFF');
        string[] headers = SplitCsvLine(headerLine);
        var index = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int h = 0; h < headers.Length; h++)
        {
            string key = headers[h].Trim().TrimStart('\uFEFF');
            if (key.Length > 0)
                index[key] = h;
        }

        Debug.Log($"[PoseCsvAutoCapture] Loading CSV: {csvPath}");
        Debug.Log($"[PoseCsvAutoCapture] Headers({headers.Length}): {string.Join(" | ", index.Keys)}");

        // image_file (tag CSV) or photo (real-photo CSV)
        int imageCol = FindCol(index, "image_file", "photo");
        if (imageCol < 0)
            throw new Exception("CSV missing column: image_file (or photo)");

        Require(index, "unity_pos_x");
        Require(index, "unity_pos_y");
        Require(index, "unity_pos_z");
        Require(index, "unity_rot_x");
        Require(index, "unity_rot_y");
        Require(index, "unity_rot_z");

        // Input CSV columns are named unity_quat_*; export file renames them to csv_quat_* for comparison.
        int qxCol = FindCol(index, "unity_quat_x", "csv_quat_x");
        int qyCol = FindCol(index, "unity_quat_y", "csv_quat_y");
        int qzCol = FindCol(index, "unity_quat_z", "csv_quat_z");
        int qwCol = FindCol(index, "unity_quat_w", "csv_quat_w");
        bool hasQuatCols = qxCol >= 0 && qyCol >= 0 && qzCol >= 0 && qwCol >= 0;

        if (!hasQuatCols)
        {
            Debug.LogWarning(
                "[PoseCsvAutoCapture] No quaternion columns found " +
                "(need unity_quat_x/y/z/w). Will apply euler only; " +
                "export csv_quat_* will be 0. Check csvRelativePath points to a " +
                "converted pose CSV that contains those columns.");
        }
        else
        {
            Debug.Log("[PoseCsvAutoCapture] Quaternion columns found — will fill csv_quat_* in export.");
        }

        for (int i = 1; i < lines.Length; i++)
        {
            string line = lines[i].Trim();
            if (string.IsNullOrEmpty(line))
                continue;

            string[] cols = SplitCsvLine(line);
            string imageFile = cols[imageCol].Trim();
            // Keep only file name if a full path was stored
            imageFile = Path.GetFileName(imageFile);
            // real-photo CSV may use bare ids like "Baseline" / "1"
            if (!imageFile.Contains("."))
                imageFile = imageFile + ".jpg";

            var pose = new PoseRow
            {
                imageFile = imageFile,
                position = new Vector3(
                    ParseFloat(cols[index["unity_pos_x"]]),
                    ParseFloat(cols[index["unity_pos_y"]]),
                    ParseFloat(cols[index["unity_pos_z"]])
                ),
                euler = new Vector3(
                    ParseFloat(cols[index["unity_rot_x"]]),
                    ParseFloat(cols[index["unity_rot_y"]]),
                    ParseFloat(cols[index["unity_rot_z"]])
                ),
                hasQuat = false,
            };

            if (hasQuatCols)
            {
                if (cols.Length <= Mathf.Max(qxCol, Mathf.Max(qyCol, Mathf.Max(qzCol, qwCol))))
                {
                    Debug.LogError($"Row {i} has only {cols.Length} columns; cannot read quaternion.");
                }
                else
                {
                    float qx = ParseFloat(cols[qxCol]);
                    float qy = ParseFloat(cols[qyCol]);
                    float qz = ParseFloat(cols[qzCol]);
                    float qw = ParseFloat(cols[qwCol]);
                    if (qw < 0f)
                    {
                        qx = -qx; qy = -qy; qz = -qz; qw = -qw;
                    }
                    pose.hasQuat = true;
                    pose.csvQuatX = qx;
                    pose.csvQuatY = qy;
                    pose.csvQuatZ = qz;
                    pose.csvQuatW = qw;
                    pose.quat = new Quaternion(qx, qy, qz, qw);
                }
            }

            rows.Add(pose);
        }

        int withQuat = 0;
        for (int i = 0; i < rows.Count; i++)
            if (rows[i].hasQuat) withQuat++;
        Debug.Log($"[PoseCsvAutoCapture] Loaded {rows.Count} rows, {withQuat} with quaternion.");
        if (rows.Count > 0)
        {
            var r0 = rows[0];
            Debug.Log(
                $"[PoseCsvAutoCapture] Row0 {r0.imageFile} euler=({r0.euler.x:F3},{r0.euler.y:F3},{r0.euler.z:F3}) " +
                $"quat=({r0.csvQuatX:F4},{r0.csvQuatY:F4},{r0.csvQuatZ:F4},{r0.csvQuatW:F4}) hasQuat={r0.hasQuat}");
        }

        return rows;
    }

    private static int FindCol(Dictionary<string, int> index, params string[] names)
    {
        for (int i = 0; i < names.Length; i++)
        {
            if (index.TryGetValue(names[i], out int col))
                return col;
        }
        return -1;
    }

    private static void Require(Dictionary<string, int> index, string key)
    {
        if (!index.ContainsKey(key))
            throw new Exception($"CSV missing column: {key}");
    }

    private static float ParseFloat(string s)
    {
        return float.Parse(s.Trim(), CultureInfo.InvariantCulture);
    }

    private static string[] SplitCsvLine(string line)
    {
        // Simple CSV split (no quoted commas in this file).
        return line.Split(',');
    }

    private void CaptureRGB(string baseName)
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

        // JPG to match CamCoordTest_*.jpg naming style; also write PNG if you prefer lossless.
        string filename = Path.Combine(saveDirectory, baseName + ".jpg");
        File.WriteAllBytes(filename, screenshot.EncodeToJPG(95));

        Destroy(screenshot);
        Destroy(rt);
        Debug.Log($"RGB saved: {filename}");
    }

    private void CaptureDepth(string baseName)
    {
        if (depthMaterial == null)
        {
            Debug.LogError("Depth Material 未赋值！请在 Inspector 中指定（与 DepthCalibration 相同）。");
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

        string filename = Path.Combine(saveDirectory, baseName + ".png");
        File.WriteAllBytes(filename, tex.EncodeToPNG());

        Destroy(tex);
        Debug.Log($"Depth saved: {filename}");
    }
}
