using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MagnetHe : MonoBehaviour
{
    public float speed;
    private Rigidbody rb;

    // 螺旋扫描参数
    public bool isScanning = false; // 是否启用螺旋扫描
    public float scanRadius = 3.5f; // 扫描半径
    public float v_z = 1.0f; // Z方向扫描速度
    public float v_theta = 90.0f; // 角度变化速度(度/秒)
    public Vector3 scanAxis = new Vector3(0.662f, 9.12f, 0f); // 扫描轴点
    public float zMin = -3.2f; // Z方向最小值
    public float zMax = -1.6f; // Z方向最大值
    public float screenshotInterval = 4.0f; // 更新间隔
    private float currentTheta = 0f; // 当前角度
    private float currentZ = 0f; // 当前Z坐标
    private bool movingDown = true; // Z方向移动标志
    private float timer;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        
        // 设置初始方向为欧拉角 (0, 0, 0)
        transform.rotation = Quaternion.Euler(180, 0, 0);

        // 设置初始位置为 scanAxis - (scanRadius, 0, -zMax)
        transform.position = scanAxis - new Vector3(scanRadius, 0, -zMax);
        currentZ = transform.position.z; // 初始化当前Z坐标
        timer = 0f; // 初始化计时器
    }
    void FixedUpdate()
    {
        // 检测空格键按下，开始/停止扫描
        if (Input.GetKeyDown(KeyCode.Space))
        {
            isScanning = !isScanning;

            if (isScanning)
            {
                currentZ = transform.position.z; // 记录开始扫描时的Z坐标
                currentTheta = 0f; // 重置角度
                Debug.Log($"[螺旋扫描] 已启动 - 初始Z坐标: {currentZ:F2}, 扫描半径: {scanRadius:F2}");
            }
            else
            {
                Debug.Log($"[螺旋扫描] 已停止 - 最终Z坐标: {transform.position.z:F2}, 总旋转角度: {currentTheta:F2}°");
            }
        }

        // 执行螺旋扫描
        if (isScanning)
        {
            timer += Time.fixedDeltaTime; // 增加计时器

            if (timer >= screenshotInterval)
            {
                PerformHelicalScan(); // 执行扫描
                timer = 0f; // 重置计时器
            }
        }
    }


    void PerformHelicalScan()
    {
        // 更新角度
        currentTheta += v_theta * screenshotInterval; // 使用间隔时间更新角度
        currentTheta = currentTheta % 360; // 保持在0-360度之间

        // 更新Z坐标
        if (movingDown)
        {
            currentZ -= v_z * screenshotInterval; // 使用间隔时间更新Z坐标
            if (currentZ <= zMin)
            {
                currentZ = zMin;
                movingDown = false;
            }
        }
        else
        {
            currentZ += v_z * screenshotInterval; // 使用间隔时间更新Z坐标
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
            scanAxis.x - xOffset,
            scanAxis.y - yOffset,
            currentZ
        );

        transform.position = newPosition;

        // 使物体始终指向扫描轴点
        Vector3 directionToAxis = scanAxis - transform.position;
        directionToAxis.z = 0; // 只在XY平面旋转

        // 确保方向不为零，避免无效旋转
        if (directionToAxis != Vector3.zero)
        {
            // 计算目标旋转
            Quaternion targetRotation = Quaternion.LookRotation(Vector3.forward, directionToAxis.normalized);

            // 设置物体的旋转，使其与 Y 轴垂直
            transform.rotation = Quaternion.Euler(0, 0, targetRotation.eulerAngles.z);
        }
    }
}