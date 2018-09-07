using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

/// <summary>
/// Represents a sample from the mocap system, without any sort of post-processing.
/// All positions are in world space and can be missing (i.e. set to NaN).
/// </summary>
public struct MocapSample : IComparable<MocapSample>
{
    private static int maxMarkers = 9;

    public float time;

    public Vector3[] position;
    public bool[] isValid;

    public MocapSample(float time, int numMarkers)
    {
        this.time = time;
        this.position = new Vector3[numMarkers];
        this.isValid = new bool[numMarkers];
    }

    public MocapSample(string input)
    {
        var strings = input.Split(',');
        int numMarkers = Math.Min(maxMarkers, (strings.Length - 1) / 4);
        position = new Vector3[numMarkers];
        isValid = new bool[numMarkers];

        time = float.Parse(strings[0], CultureInfo.InvariantCulture);
        int i = 1;
        int marker = 0;
        while (i < strings.Length && marker < maxMarkers)
        {
            position[marker].x = float.Parse(strings[i++], CultureInfo.InvariantCulture);
            position[marker].y = float.Parse(strings[i++], CultureInfo.InvariantCulture);
            position[marker].z = float.Parse(strings[i++], CultureInfo.InvariantCulture);
            isValid[marker] = int.Parse(strings[i++]) == 1;

            marker++;
        }
    }

    public static void Export(string csvPath, IEnumerable<MocapSample> samples, IEnumerable<string> markerNames)
    {
        using (var file = new StreamWriter(csvPath))
        {
            // Write header
            var features = new List<string>();
            features.Add("time");
            foreach (string name in markerNames)
            {
                features.Add(name + ".pos.x");
                features.Add(name + ".pos.y");
                features.Add(name + ".pos.z");
                features.Add(name + ".isValid");
            }
            file.WriteLine(string.Join(",", features.ToArray()));

            // Write records
            foreach (var s in samples)
            {
                file.WriteLine(s);
            }
        }
    }

    public static List<MocapSample> Import(string csvPath)
    {
        var output = new List<MocapSample>();
        using (var file = new StreamReader(csvPath))
        {
            string line;
            while ((line = file.ReadLine()) != null)
            {
                if (!line.StartsWith("time")) // Skip header
                {
                    output.Add(new MocapSample(line));
                }
            }
        }
        return output;
    }

    public override string ToString()
    {
        var strings = new List<string>();
        strings.Add(time.ToString(CultureInfo.InvariantCulture));
        for (int i = 0; i < position.Length; i++)
        {
            strings.Add(position[i].x.ToString(CultureInfo.InvariantCulture));
            strings.Add(position[i].y.ToString(CultureInfo.InvariantCulture));
            strings.Add(position[i].z.ToString(CultureInfo.InvariantCulture));
            strings.Add((isValid[i] ? 1 : 0).ToString());
        }
        return string.Join(",", strings.ToArray());
    }

    public int CompareTo(MocapSample other)
    {
        return time.CompareTo(other.time);
    }
}
