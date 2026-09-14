using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallMagnetWave : MonoBehaviour
{
    public float speed;
    private Rigidbody rb;

    // 螺旋扫描参数
    public bool isScanning = false; // 是否启用螺旋扫描
    public float scanRadius = 1.5f; // 扫描半径
    public float v_z = 0.1f; // Z方向扫描速度
    public float v_theta = 90.0f; // 角度变化速度(度/秒)

    public Vector3 scanAxis = new Vector3(0.662f, 9.12f, 0f); // 扫描轴点
    public float zMin = -3.2f; // Z方向最小值
    public float zMax = -1.6f; // Z方向最大值


    private bool canScan=false;
    private float currentTheta = 0f; // 当前角度
    private float currentZphase = 0f;
    private float currentZ = 0f; // 当前Z坐标
    private bool movingDown = true; // Z方向移动标志

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        // 设置初始方向为欧拉角 (0, 0, 0)
        transform.rotation = Quaternion.Euler(0, 90, 90);

        // 设置初始位置为 scanAxis - (scanRadius, 0, -zMax)
        transform.position = scanAxis - new Vector3(-scanRadius, 0, -zMin);

        currentZ = transform.position.z; // 初始化当前Z坐标
        // Start the delay coroutine automatically
        StartCoroutine(DelayedWaveScan());
    }

    void FixedUpdate()
    {
        // 保留原有的键盘控制代码
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            Vector3 position = this.transform.position;
            position.x = position.x - 0.01f * speed;
            this.transform.position = position;
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            Vector3 position = this.transform.position;
            position.x = position.x + 0.01f * speed;
            this.transform.position = position;
        }
        if (Input.GetKey(KeyCode.UpArrow))
        {
            Vector3 position = this.transform.position;
            position.z = position.z + 0.01f * speed;
            this.transform.position = position;
        }
        if (Input.GetKey(KeyCode.DownArrow))
        {
            Vector3 position = this.transform.position;
            position.z = position.z - 0.01f * speed;
            this.transform.position = position;
        }

        if (Input.GetKey(KeyCode.W))
        {
            Vector3 position = this.transform.position;
            position.y = position.y + 0.01f * speed;
            this.transform.position = position;
        }
        if (Input.GetKey(KeyCode.S))
        {
            Vector3 position = this.transform.position;
            position.y = position.y - 0.01f * speed;
            this.transform.position = position;
        }

        // 螺旋扫描控制 - 按空格键开始/停止扫描
        if (Input.GetKeyDown(KeyCode.Space))
        {
            isScanning = !isScanning;
            if (isScanning)
            {
                currentZ = transform.position.z; // 记录开始扫描时的Z坐标
                currentTheta = 0f; // 重置角度
                
            }
        }

        // 执行螺旋扫描
        if (isScanning&&canScan)
        {

            PerformWaveScan();
        }
    }
    IEnumerator DelayedWaveScan()
    {
        canScan = false;
        yield return new WaitForSeconds(10f);
        canScan = true;
    }
    void PerformWaveScan()
    {
        // Update angle for circular motion around the cylinder
        currentTheta += v_theta * Time.fixedDeltaTime;
        currentTheta = currentTheta % 360; // Keep angle between 0-360 degrees
    
        // Calculate Z position using a sine wave oscillating between zMin and zMax
        float zAmplitude = (zMax - zMin) / 2f; // Half the range for sine wave amplitude
        float zCenter = (zMax + zMin) / 2f;    // Center of the Z range
        currentZphase+=Time.fixedDeltaTime* v_z;
        currentZ = zCenter - zAmplitude * Mathf.Cos(currentZphase); // Sine wave scaled to Z range
    
        // Calculate X and Y offsets based on angle (on cylindrical surface)
        float radians = currentTheta * Mathf.Deg2Rad;
        float xOffset = scanRadius * Mathf.Cos(radians);
        float yOffset = scanRadius * Mathf.Sin(radians);
    
        // Set new position
        Vector3 newPosition = new Vector3(
            scanAxis.x + xOffset,
            scanAxis.y + yOffset,
            currentZ
        );
    
        transform.position = newPosition;
    
        // Orient object to point toward the cylinder's axis
        Vector3 directionToAxis = scanAxis - transform.position;
        directionToAxis.z = 0; // Restrict rotation to XY plane
    
        // Ensure direction is non-zero to avoid invalid rotation
        if (directionToAxis != Vector3.zero)
        {
            // Calculate target rotation to point toward axis
            Quaternion targetRotation = Quaternion.LookRotation(Vector3.forward, directionToAxis.normalized);
    
            // Apply rotation to align with target direction
            transform.rotation = targetRotation;
    
            // Rotate 90 degrees around local Y-axis to adjust orientation
            transform.Rotate(90, 0, 0, Space.Self);
        }
    }
}
