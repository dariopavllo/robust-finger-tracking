using System;

public static class Constants
{
    public const string ElbowBone = "l_elbow";
    public const string WristBone = "l_wrist";

    /* Marker IDs description:
        * 0 - "LEHAR" - Pinky base
        * 1 - "LEHAL" - Wrist
        * 2 - "WHARHAR" - Index base
        * 3 - "LRIF" - Ring fingertip
        * 4 - "LBAF" - Pinky fingertip
        * 5 - "LMIF" - Index fingertip
        * 6 - "LINF" - Middle fingertip
        * 7 - "LPAL" - Thumb 2nd joint
        * 8 - "LTHF" - Thumb fingertip
        */

    // The joints corresponding to fingertips (used for sensor fusion / post-processing)
    public static readonly string[] FingerTips = { "l_ring_tip", "l_pinky_tip", "l_index_tip", "l_middle_tip", "l_thumb_tip" };

    // Joints to export to the sensor fusion dataset
    public static readonly string[] NodesToExport = { "l_elbow", "l_wrist",
        "l_index0", "l_index1", "l_index2", "l_index3", "l_index_tip",
        "l_middle0", "l_middle1", "l_middle2", "l_middle3", "l_middle_tip",
        "l_pinky0", "l_pinky1", "l_pinky2", "l_pinky3", "l_pinky_tip",
        "l_ring0", "l_ring1", "l_ring2", "l_ring3", "l_ring_tip",
        "l_thumb1", "l_thumb2", "l_thumb3", "l_thumb_tip"
    };

    public const string OutputDirectory = "output";
}