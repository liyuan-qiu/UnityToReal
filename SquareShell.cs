using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class SquareShell : MonoBehaviour
{
    public float size = 1f;        // Outer size of the square shell (width, height, depth)
    public float thickness = 0.1f; // Thickness of the walls

    void Start()
    {
        GenerateShell();
    }

    void GenerateShell()
    {
        Mesh mesh = new Mesh();
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        meshFilter.mesh = mesh;

        float halfSize = size / 2f;
        float innerHalfSize = halfSize - thickness;

        // Vertices for a hollow square shell (front, back, left, right walls)
        Vector3[] vertices = new Vector3[]
        {
            // Front face (outer)
            new Vector3(-halfSize, -halfSize, -halfSize), // 0
            new Vector3(halfSize, -halfSize, -halfSize),  // 1
            new Vector3(halfSize, halfSize, -halfSize),   // 2
            new Vector3(-halfSize, halfSize, -halfSize),  // 3
            // Front face (inner)
            new Vector3(-innerHalfSize, -innerHalfSize, -halfSize), // 4
            new Vector3(innerHalfSize, -innerHalfSize, -halfSize),  // 5
            new Vector3(innerHalfSize, halfSize, -halfSize),       // 6
            new Vector3(-innerHalfSize, halfSize, -halfSize),      // 7
            
            // Back face (outer)
            new Vector3(-halfSize, -halfSize, halfSize),  // 8
            new Vector3(halfSize, -halfSize, halfSize),   // 9
            new Vector3(halfSize, halfSize, halfSize),    // 10
            new Vector3(-halfSize, halfSize, halfSize),   // 11
            // Back face (inner)
            new Vector3(-innerHalfSize, -innerHalfSize, halfSize), // 12
            new Vector3(innerHalfSize, -innerHalfSize, halfSize),  // 13
            new Vector3(innerHalfSize, halfSize, halfSize),       // 14
            new Vector3(-innerHalfSize, halfSize, halfSize)       // 15
        };

        // Triangles for front, back, left, right walls
        int[] triangles = new int[]
        {
            // Front face (outer and inner)
            0, 2, 1,  0, 3, 2,
            4, 6, 5,  4, 7, 6,
            // Front edges (top, bottom, left, right)
            3, 7, 2,  2, 7, 6,
            0, 1, 4,  1, 5, 4,
            0, 4, 3,  3, 4, 7,
            1, 2, 5,  2, 6, 5,
            // Back face (outer and inner)
            9, 10, 8,  10, 11, 8, // Reversed winding for back face
            12, 14, 13,  12, 15, 14,
            // Back edges
            11, 10, 15,  10, 14, 15,
            8, 9, 12,  9, 13, 12,
            8, 12, 11,  11, 15, 12,
            9, 10, 13,  10, 14, 13,
            // Left face
            0, 8, 3,  3, 8, 11,
            4, 12, 7,  7, 12, 15,
            // Right face
            1, 2, 9,  2, 10, 9,
            5, 6, 13,  6, 14, 13
        };

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals(); // Ensures proper lighting

        // Assign material
        MeshRenderer renderer = GetComponent<MeshRenderer>();
        Material material;

        // Use your depth shader or a default HDRP material
        if (Shader.Find("Custom/HDRPDepthShader2019") != null)
        {
            material = new Material(Shader.Find("Custom/HDRPDepthShader2019"));
        }
        else
        {
            material = new Material(Shader.Find("HDRP/Lit"));
            Debug.LogWarning("Custom shader not found. Using HDRP/Lit instead.");
        }

        renderer.material = material;
    }
}
