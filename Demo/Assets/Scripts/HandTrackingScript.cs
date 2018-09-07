using UnityEngine;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using HandTracking;

public class HandTrackingScript : MonoBehaviour
{
    // Aliases of the markers
    public string[] markerIds;

    // Whether to draw or not the markers
    public bool drawMarkers = true;

    private MarkerUI markerUI;

    // The marker positions at the current frame (alias -> position map)
    private Dictionary<string, Vector3> markerPositions;

    // Lists of recorded or loaded data
    private List<MocapSample> mocapCaptureData;
    private List<FusedSample> fusedCaptureData;

    private bool playback = false;
    private bool jointPredictions = false;
    private int predictionMode = 0;
    private bool postProcessing = true;
    private bool pauseKeyFrame = false;
    private int lastKeyFrame = -1;
    private float startTime;

    private OcclusionManager occlusionManager;
    private HandTemplate handTemplate;
    private JointPredictor jointPredictor;

    void Awake()
    {
        UnityEngine.VR.VRSettings.enabled = false;
    }

    void Start()
    {
        markerUI = new MarkerUI();

        markerPositions = new Dictionary<string, Vector3>();

        // This one initilize the render marker to render the list of trackers written in the file alias_marker
        if (drawMarkers)
        {
            markerUI.InitMarkerRendering();
        }

        mocapCaptureData = new List<MocapSample>();
        fusedCaptureData = new List<FusedSample>();

        handTemplate = new HandTemplate();

        /* Marker IDs description:
         * 0 - Pinky base
         * 1 - Wrist
         * 2 - Index base
         * 3 - Ring fingertip
         * 4 - Pinky fingertip
         * 5 - Index fingertip
         * 6 - Middle fingertip
         * 7 - Thumb 2nd joint
         * 8 - Thumb fingertip
         */

        /*
         * Specify the offsets for the alignment points. For instance, if the marker corresponding to the wrist is 3 cm above the wrist bone, you should define an offset of (-0.03, 0, 0).
         * This is not strictly necessary, but it helps with improving the precision (and allows the markers to be placed in arbitrary locations).
         * All coordinates are in object space.
         */

        handTemplate.AddMarker("LEHAR", "l_pinky1", new Vector3(-0.01f, 0.0f, 0.0f), true);
        handTemplate.AddMarker("LEHAL", "l_wrist", new Vector3(-0.03f, 0.0f, 0.0f), true);
        handTemplate.AddMarker("WHARHAR", "l_index1", new Vector3(-0.01f, 0.0f, 0.0f), true);
        handTemplate.AddMarker("LRIF", "l_ring_tip", Vector3.zero);
        handTemplate.AddMarker("LBAF", "l_pinky_tip", Vector3.zero);
        handTemplate.AddMarker("LMIF", "l_index_tip", Vector3.zero);
        handTemplate.AddMarker("LINF", "l_middle_tip", Vector3.zero);
        handTemplate.AddMarker("LPAL", "l_thumb2", new Vector3(-0.01f, 0.0f, 0.0f));
        handTemplate.AddMarker("LTHF", "l_thumb_tip", Vector3.zero);

        handTemplate.SetPivot("l_wrist");
        handTemplate.SaveToFile(Constants.OutputDirectory + "\\hand.csv");

        foreach (var marker in handTemplate.MarkerList)
        {
            markerPositions[marker] = handTemplate.GetInitialMarkerWorldPosition(marker);
            int j = markerUI.marker_alias_to_ids[marker];
            markerUI.marker_positions[j, 0] = -markerPositions[marker].x;
            markerUI.marker_positions[j, 1] = markerPositions[marker].y;
            markerUI.marker_positions[j, 2] = markerPositions[marker].z;
        }

        markerUI.RenderMarkers();

        jointPredictor = new JointPredictor("models/joint_model.bin", handTemplate);

        jointPredictor.AddJoint("l_index1", new Vector3(1, 1, 1));
        jointPredictor.AddJoint("l_index2", new Vector3(0, 1, 0));
        jointPredictor.AddJoint("l_index3", new Vector3(0, 1, 0));
        jointPredictor.AddJoint("l_middle1", new Vector3(1, 1, 1));
        jointPredictor.AddJoint("l_middle2", new Vector3(0, 1, 0));
        jointPredictor.AddJoint("l_middle3", new Vector3(0, 1, 0));
        jointPredictor.AddJoint("l_pinky1", new Vector3(1, 1, 1));
        jointPredictor.AddJoint("l_pinky2", new Vector3(0, 1, 0));
        jointPredictor.AddJoint("l_pinky3", new Vector3(0, 1, 0));
        jointPredictor.AddJoint("l_ring1", new Vector3(1, 1, 1));
        jointPredictor.AddJoint("l_ring2", new Vector3(0, 1, 0));
        jointPredictor.AddJoint("l_ring3", new Vector3(0, 1, 0));
        jointPredictor.AddJoint("l_thumb1", new Vector3(1, 1, 1));
        jointPredictor.AddJoint("l_thumb2", new Vector3(1, 1, 0));
        jointPredictor.AddJoint("l_thumb3", new Vector3(1, 0, 0));


        occlusionManager = new OcclusionManager(handTemplate.MarkerList);
        occlusionManager.AddImplementation(new NaiveOcclusionPredictor());
    }

    void OnGUI()
    {
        if (!playback)
        {
            if (GUILayout.Button("Play recording"))
            {
                string path = UnityEditor.EditorUtility.OpenFilePanel("Select dump", Constants.OutputDirectory, "csv");
                if (path != null && path.Length > 0)
                {
                    path = path.Replace("mocap", "***").Replace("neuron", "***").Replace("fused", "***");

                    Debug.Log("Loading recording from disk...");

                    mocapCaptureData = MocapSample.Import(path.Replace("***", "mocap"));
                    fusedCaptureData = FusedSample.Import(path.Replace("***", "fused"));

                    Debug.Log("Playing...");
                    playback = true;
                    startTime = Time.time;
                }
            }
        }
        else if (playback)
        {
            if (GUILayout.Button("Stop playback"))
            {
                Debug.Log("Stopped.");
                playback = false;
            }

            pauseKeyFrame = GUILayout.Toggle(pauseKeyFrame, "Pause current key frame");
        }

        if (playback)
        {
            float time;
            int keyFrame;
            keyFrame = GetCurrentKeyFrame();
            time = mocapCaptureData[keyFrame].time - mocapCaptureData[0].time;
            GUILayout.Label("Time: " + Math.Round(time, 2) + " s");
            GUILayout.Label("Key frame: " + keyFrame);
        }

        drawMarkers = GUILayout.Toggle(drawMarkers, "Draw markers");
        if (!drawMarkers)
        {
            for (int i = 0; i < markerUI.marker_positions.GetLength(0); i++)
            {
                for (int j = 0; j < markerUI.marker_positions.GetLength(1); j++)
                {
                    markerUI.marker_positions[i, j] = -1000;
                }
            }
            markerUI.RenderMarkers();
        }

        jointPredictions = GUILayout.Toggle(jointPredictions, "Prediction mode");
        if (!jointPredictions)
        {
            GUILayout.Label("Displaying raw dataset");
        }
        else
        {
            GUILayout.Label("Displaying NN predictions");
        }

        if (jointPredictions)
        {
            GUILayout.Label("Occlusion prediction:");
            bool opt0 = predictionMode == 0;
            bool opt1 = predictionMode == 1;
            bool opt2 = predictionMode == 2;
            bool opt3 = predictionMode == 3;
            opt0 = GUILayout.Toggle(opt0, "None (last position)");
            opt1 = GUILayout.Toggle(opt1, "Moving average");
            opt2 = GUILayout.Toggle(opt2, "Affine combination");
            opt3 = GUILayout.Toggle(opt3, "Neural network");
            int newPredictionMode = 0;
            if (opt0 && predictionMode != 0)
            {
                newPredictionMode = 0;
            }
            else if (opt1 && predictionMode != 1)
            {
                newPredictionMode = 1;
            }
            else if (opt2 && predictionMode != 2)
            {
                newPredictionMode = 2;
            }
            else if (opt3 && predictionMode != 3)
            {
                newPredictionMode = 3;
            }
            else
            {
                newPredictionMode = predictionMode;
            }

            if (newPredictionMode != predictionMode)
            {
                occlusionManager.RemoveAllImplementations();
                switch (newPredictionMode)
                {
                    case 0:
                        occlusionManager.AddImplementation(new NaiveOcclusionPredictor());
                        break;
                    case 1:
                        occlusionManager.AddImplementation(new MovingAverageOcclusionPredictor(20, handTemplate));
                        break;
                    case 2:
                        occlusionManager.AddImplementation(new LinearOcclusionPredictor());
                        break;
                    case 3:
                        occlusionManager.AddImplementation(new NeuralOcclusionPredictor("models/marker_model.bin", handTemplate, true, true));
                        break;
                }
            }
            predictionMode = newPredictionMode;

            if (predictionMode > 0)
            {
                GUILayout.Label("");
                postProcessing = GUILayout.Toggle(postProcessing, "Post-processing");
            }

            GUILayout.Label("Use keyboard [1-9]\nto simulate occlusions.");
        }
    }

    private int GetCurrentKeyFrame()
    {
        float startTime = mocapCaptureData[0].time;
        float endTime = mocapCaptureData[mocapCaptureData.Count - 1].time;
        float wrappedTime = (Time.time - this.startTime) % (endTime - startTime) + startTime;
        int keyFrame = mocapCaptureData.BinarySearch(new MocapSample(wrappedTime, markerPositions.Count));
        if (keyFrame == -1)
        {
            keyFrame = 0;
        }
        if (keyFrame < 0)
        {
            keyFrame = (~keyFrame) - 1;
        }

        return keyFrame;
    }

    // Update is called once per frame
    void Update()
    {
        var time = Time.time - startTime;
        int keyFrame = -1;
        if (playback)
        {
            keyFrame = GetCurrentKeyFrame();
            if (pauseKeyFrame)
            {
                keyFrame = lastKeyFrame;
            }
            lastKeyFrame = keyFrame;

            //This one allows to get an array of vectors positions of the trackers wanted from PhaseSpace
            for (int i = 0; i < markerIds.Length; i++)
            {
                string id = markerIds[i];
                int j = markerUI.marker_alias_to_ids[id];

                if (!float.IsNaN(mocapCaptureData[keyFrame].position[i].x))
                {
                    markerPositions[id] = mocapCaptureData[keyFrame].position[i];
                    markerUI.marker_positions[j, 0] = -markerPositions[markerIds[i]].x;
                    markerUI.marker_positions[j, 1] = markerPositions[markerIds[i]].y;
                    markerUI.marker_positions[j, 2] = markerPositions[markerIds[i]].z;
                }
                else
                {
                    markerPositions[id] = mocapCaptureData[keyFrame].position[i];
                    markerUI.marker_positions[j, 0] = float.NaN;
                    markerUI.marker_positions[j, 1] = float.NaN;
                    markerUI.marker_positions[j, 2] = float.NaN;
                }
            }

        }
        else
        {
            mocapCaptureData.Clear();

            keyFrame = mocapCaptureData.Count;
            for (int i = 0; i < markerIds.Length; i++)
            {
                Vector3 pos;
                markerUI.GetMarkerPosition(markerIds[i], out pos);
                markerPositions[markerIds[i]] = pos;
            }

            MocapSample sM = new MocapSample(time, markerPositions.Count);
            for (int i = 0; i < handTemplate.MarkerList.Count; i++)
            {
                sM.position[i] = markerPositions[handTemplate.MarkerList[i]];
                sM.isValid[i] = !float.IsNaN(markerPositions[handTemplate.MarkerList[i]].x);
            }
            mocapCaptureData.Add(sM);
        }

        for (int i = 0; i < markerIds.Length; i++)
        {
            // Keyboard buttons [0-9] simulate sensor occlusion
            bool overrideMarker = Input.GetKey((KeyCode)((int)KeyCode.Alpha1 + i));
            if (overrideMarker)
            {
                string id = markerIds[i];
                int j = markerUI.marker_alias_to_ids[id];

                markerPositions[id] = new Vector3(float.NaN, float.NaN, float.NaN);
                markerUI.marker_positions[j, 0] = float.NaN;
                markerUI.marker_positions[j, 1] = float.NaN;
                markerUI.marker_positions[j, 2] = float.NaN;
            }
        }


        // Reset initial orientations
        foreach (var m in Constants.NodesToExport)
        {
            var transform = GameObject.Find(m).transform;
            transform.localRotation = Quaternion.identity;
        }

        if (!playback)
        {
            // No input data available
            return;
        }

        occlusionManager.Update(markerPositions, Time.deltaTime);

        if (!jointPredictions)
        {
            FusedSample s = fusedCaptureData[keyFrame];
            for (int i = 0; i < s.position.Length; i++)
            {
                var obj = GameObject.Find(Constants.NodesToExport[i]);
                Vector3 rot = s.orientation[i];

                obj.transform.localRotation = Quaternion.Euler(rot);
            }
        }
        else
        {
            var output = jointPredictor.Predict(markerPositions);
            foreach (var joint in output)
            {
                GameObject.Find(joint.Key).transform.localRotation = joint.Value;
            }
        }

        List<int> nanIndices = new List<int>();
        foreach (var marker in markerPositions)
        {
            int j = markerUI.marker_alias_to_ids[marker.Key];
            if (double.IsNaN(markerUI.marker_positions[j, 0]))
            {
                nanIndices.Add(j);
            }
            markerUI.marker_positions[j, 0] = -marker.Value.x;
            markerUI.marker_positions[j, 1] = marker.Value.y;
            markerUI.marker_positions[j, 2] = marker.Value.z;
        }

        if (drawMarkers)
        {
            markerUI.RenderMarkers();
        }

        foreach (int i in nanIndices)
        {
            markerUI.marker_spheres[i].GetComponent<Renderer>().material.color = Color.red;
        }

        // Pose estimation: move the hand template towards the markers
        var X = new List<Vector3>();
        var Y = new List<Vector3>();
        foreach (var markerId in handTemplate.AlignmentMarkerList)
        {
            Transform tBone = GameObject.Find(handTemplate.GetJointName(markerId)).transform;
            var pos = tBone.TransformPoint(handTemplate.GetMarkerOffset(markerId)); // Add offset
            X.Add(pos);

            var posTarget = markerPositions[markerId];
            Y.Add(posTarget);
        }

        double[,] R;
        double[] offset;
        HandTracking.Utils.ComputeRigidMotion(X, Y, out R, out offset);
        HandTracking.Utils.RotateAndShift(GameObject.Find(Constants.ElbowBone).transform, R, offset);

        // Post-processing: align finger directions
        if (postProcessing && jointPredictions)
        {
            for (int i = 0; i < Constants.FingerTips.Length; i++)
            {
                var tipJoint = GameObject.Find(Constants.FingerTips[i]).transform;
                var baseJoint = tipJoint.parent.parent.parent;
                Vector3 targetPosition = markerPositions[handTemplate.GetMarkerId(Constants.FingerTips[i])];
                if (!float.IsNaN(targetPosition.x))
                {
                    HandTracking.Utils.AlignDirections(baseJoint, tipJoint.position, targetPosition);
                }
            }
        }

        // Focus on the hand
        Camera.main.transform.LookAt(GameObject.Find(Constants.WristBone).transform.position);

        float scroll = Input.GetAxis("Mouse ScrollWheel");
        Camera.main.fieldOfView -= scroll*10;
    }
}