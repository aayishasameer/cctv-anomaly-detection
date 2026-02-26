#!/bin/bash
# Monitor processing progress

echo "🔍 Monitoring CCTV Detection Progress"
echo "======================================"
echo ""

# Check if output file exists and is growing
if [ -f "shoplifting_detection_output.mp4" ]; then
    SIZE=$(du -h shoplifting_detection_output.mp4 | cut -f1)
    echo "📹 Output file: shoplifting_detection_output.mp4"
    echo "📊 Current size: $SIZE"
else
    echo "⏳ Output file not created yet..."
fi

echo ""
echo "💡 The system is processing the video in the background"
echo "💡 This will take approximately 5-10 minutes"
echo ""
echo "To check detailed progress, run:"
echo "   tail -f /tmp/cctv_processing.log"
