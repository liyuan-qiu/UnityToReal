using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MagnetHellicallyScan : MonoBehaviour
{
    public float speed;
    private Rigidbody rb;

    // 螺旋扫描参数
    public bool isScanning = false; // 是否启用螺旋扫描
    public float scanRadius = 1.5f; // 扫描半径
    public float v_z = 1.0f; // Z方向扫描速度
    public float v_theta = 90.0f; // 角度变化速度(度/秒)
    public Vector3 scanAxis = new Vector3(-1.765f, 8.957f, 0); // 扫描轴点
    public float zMin = -3.2f; // Z方向最小值
    public float zMax = -1.6f; // Z方向最大值

    private float currentTheta = 0f; // 当前角度
    private float currentZ = 0f; // 当前Z坐标
    private bool movingDown = true; // Z方向移动标志

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        currentZ = transform.position.z; // 初始化当前Z坐标
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
        if (isScanning)
        {
            PerformHelicalScan();
        }
    }

    void PerformHelicalScan()
    {
        // 更新角度
        currentTheta += v_theta * Time.fixedDeltaTime;
        currentTheta = currentTheta % 360; // 保持在0-360度之间

        // 更新Z坐标
        if (movingDown)
        {
            currentZ -= v_z * Time.fixedDeltaTime;
            if (currentZ <= zMin)
            {
                currentZ = zMin;
                movingDown = false;
            }
        }
        else
        {
            currentZ += v_z * Time.fixedDeltaTime;
            if (currentZ >= zMax)
            {
                currentZ = zMax;
                movingDown = true;
            }
        }

        // 计算基于角度的X和Y位置（相对于扫描轴点）
        float radians = currentTheta * Mathf.Deg2Rad;
        float xOffset = scanRadius * Mathf.Cos(radians);
        float yOffset = scanRadius * Mathf.Sin(radians);

        // 设置新位置
        Vector3 newPosition = new Vector3(
            scanAxis.x + xOffset,
            scanAxis.y + yOffset,
            currentZ
        );

        transform.position = newPosition;

        // 使物体始终指向扫描轴点
        Vector3 directionToAxis = scanAxis - transform.position;
        directionToAxis.z = 0; // 只在XY平面旋转

        if (directionToAxis != Vector3.zero)
        {
            transform.rotation = Quaternion.LookRotation(directionToAxis.normalized, Vector3.forward);
        }
    }
}