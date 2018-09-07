using UnityEngine;
using System;
using System.Collections.Generic;

public class MarkerUI
{
    private static string[] markerNames = { "LEHAR", "LEHAL", "WHARHAR", "LRIF", "LBAF", "LMIF", "LINF", "LPAL", "LTHF" };

    public double[,] marker_positions;
	public int[] marker_ids;
    public Dictionary<int, string> marker_ids_to_alias;
    public Dictionary<string, int> marker_alias_to_ids;
    public int no_of_markers;
	public GameObject[] marker_spheres;
		
	public MarkerUI()
	{
        marker_positions = new double[markerNames.Length, 3];
        marker_ids = new int[markerNames.Length];
        no_of_markers = markerNames.Length;
        marker_ids_to_alias = new Dictionary<int, string>();
        marker_alias_to_ids = new Dictionary<string, int>();
        for (int i = 0; i < markerNames.Length; i++)
        {
            marker_ids_to_alias[i] = markerNames[i];
            marker_alias_to_ids[markerNames[i]] = i;
            marker_ids[i] = i;
        }
	}

	public void RenderMarkers()
	{
		for (int i = 0; i < no_of_markers; ++i)
		{
            if (IsMarkerVisible(i))
            {
                marker_spheres[i].transform.position = new Vector3( (float)(-marker_positions[i, 0]),
                                                                    (float)(marker_positions[i, 1]),
                                                                    (float)(marker_positions[i, 2]));

                if (marker_spheres[i].GetComponent<Renderer>().material.color == Color.red)
                    marker_spheres[i].GetComponent<Renderer>().material.color = Color.blue;
            }
            else
            {
                marker_spheres[i].GetComponent<Renderer>().material.color = Color.red;
            }
		}
	}

	public void InitMarkerRendering()
	{
		marker_spheres = new GameObject[no_of_markers];
			
		for(int i = 0; i < no_of_markers; ++i)
		{
			marker_spheres[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
			marker_spheres[i].transform.localScale = new Vector3((float)0.02, (float)0.02, (float)0.02);
			marker_spheres[i].GetComponent<Renderer>().material.color = new Color(0, 0, 1);
			marker_spheres[i].name = marker_ids_to_alias[marker_ids[i]];
		}
	}

	public void StopMarkerRendering()
	{
		for(int i = 0; i < no_of_markers; ++i)
			GameObject.Destroy(marker_spheres[i]);
		marker_spheres = null;
	}

    public bool GetMarkerPosition(int markerID, out Vector3 position)
    {
        if (markerID < no_of_markers)
        {
            position = new Vector3((float)(-marker_positions[markerID, 0]),
                                (float)(marker_positions[markerID, 1]),
                                (float)(marker_positions[markerID, 2]));
               
        }
        else
        {
            Debug.LogWarning("ID " + markerID + " is too big, cannot find marker.");
            position = Vector3.zero;
        }

        return markerID < no_of_markers;
    }

    public bool GetMarkerPosition(string markerName, out Vector3 position)
    {
        int id;
        if (marker_alias_to_ids.TryGetValue(markerName, out id))
        {
            return GetMarkerPosition(id, out position);
        }
        else
        {
            Debug.LogWarning("Marker " + markerName + " cannot be found in VRPN client list.");
            position = Vector3.zero;
            return false;
        }
    }

    public bool IsMarkerVisible(int markerID)
    {
        return !Double.IsNaN(marker_positions[markerID, 0]);
    }

    public bool IsMarkerVisible(string markerName)
    {
        int id;
        if (marker_alias_to_ids.TryGetValue(markerName, out id))
        {
            return IsMarkerVisible(id);
        }
        else
        {
            Debug.LogWarning("Marker " + markerName + " cannot be found in VRPN client list. " +
                                "Will be considered as not visible.");

            return false;
        }
    }
}