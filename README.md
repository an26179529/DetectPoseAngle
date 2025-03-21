# Pose Analyzer

A computer vision application that analyzes human posture, measures key joint angles, and provides visual feedback. Designed for sports science, physical therapy, and fitness training applications.

## Features

- **Photo and Video Analysis**: Support for analyzing both static images and dynamic videos
- **Multi-angle Measurement**: Calculates and displays the following key angles:
  - Hip angle (angle between hip joint center and knees)
  - Body-leg angle (angle between shoulders, hips, and ankles)
- **Selective Display**: Option to show measurements for left side, right side, or both sides
- **Intuitive Interface**: Clean and user-friendly graphical user interface
- **Real-time Preview**: Visual feedback during processing
- **Progress Monitoring**: Progress bar and percentage display during video processing

## System Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- NumPy
- Tkinter (Python built-in)
- PIL (Pillow)

## Installation

1. Ensure Python 3.6 or higher is installed
2. Install required dependencies:
```bash
pip install opencv-python mediapipe numpy pillow
```

## Usage

1. Run the application:
```bash
python pose_analyzer.py
```
2. In the control panel:
   - Select input type (photo or video)
   - Choose which body side to display (left, right, or both)
   - Click "Browse" to select an input file
   - Set the output folder
   - Click "Analyze" to start processing
3. The analysis process will be displayed in the preview area, and results will be saved to the selected output folder

## Angle Measurement Details

- **Hip Angle** (red lines): Measures the angle between left knee, hip joint center, and right knee, indicating hip opening degree
- **Body-Leg Angle** (green lines): Measures the angle between shoulders, hips, and ankles, showing the relative angle between the body and legs

## Limitations and Considerations

- Analysis accuracy depends on MediaPipe's pose detection capabilities
- Loose clothing may reduce detection precision
- The person should be fully visible in the photo/video
- High-resolution video processing may require more time

## Troubleshooting

- **No pose detected**: Ensure the person is clearly visible and the pose is complete
- **Slow processing**: Consider reducing video resolution or using more powerful hardware
- **Inaccurate results**: Try capturing in well-lit environments and wearing fitted clothing

## License

This software is provided for research and educational purposes only. Not for diagnostic or treatment purposes. Users assume all risks associated with its use.

## References

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose) - Advanced pose tracking technology
- [OpenCV](https://opencv.org/) - Open source computer vision library
- [NumPy](https://numpy.org/) - Fundamental package for scientific computing with Python
- [Pillow](https://python-pillow.org/) - Python Imaging Library

## Citation

If you use this tool in your research, please cite:

```
@software{pose_analyzer,
  author = {Your Name},
  title = {Pose Analyzer: A Tool for Human Posture Analysis},
  year = {2025},
  url = {https://github.com/yourusername/pose-analyzer}
}
```

## Acknowledgments

- Google MediaPipe team for providing the pose estimation framework
- OpenCV community for computer vision tools and libraries
