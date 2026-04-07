#!/bin/bash
# Script setup ADB reverse để điện thoại dùng localhost
# Chạy mỗi lần kết nối điện thoại mới

echo "=== Setup ADB Reverse cho Backend ==="
echo ""

# Kiểm tra ADB có sẵn không
if ! command -v adb &> /dev/null; then
    echo "❌ ADB chưa được cài đặt!"
    echo "   Cài đặt: sudo apt install adb"
    exit 1
fi

# Kiểm tra device đã kết nối
echo "1. Kiểm tra thiết bị..."
DEVICES=$(adb devices | grep -v "List" | grep "device$" | wc -l)

if [ "$DEVICES" -eq 0 ]; then
    echo "❌ Không tìm thấy thiết bị!"
    echo ""
    echo "Kết nối thiết bị bằng một trong các cách:"
    echo "  - USB: Cắm cáp và bật USB Debugging"
    echo "  - Wireless: adb connect <IP>:5555"
    echo ""
    adb devices
    exit 1
fi

echo "✅ Đã tìm thấy $DEVICES thiết bị"
adb devices

# Setup port forwarding
echo ""
echo "2. Setup port forwarding..."
adb reverse tcp:8000 tcp:8000

if [ $? -eq 0 ]; then
    echo "✅ ADB reverse đã được thiết lập!"
    echo ""
    echo "=== Hoàn tất ==="
    echo "Flutter app giờ có thể dùng: http://localhost:8000"
    echo "Không cần quan tâm IP nữa!"
    echo ""
    echo "💡 Tip: Chạy script này mỗi lần kết nối điện thoại mới"
else
    echo "❌ Lỗi khi setup ADB reverse"
    exit 1
fi
