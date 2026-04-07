# 🌐 Kiến trúc Networking - Flutter (Windows) ↔ Backend (WSL)

## 🎯 Tóm tắt

**Flutter app chạy trên điện thoại Android** giao tiếp với **Backend chạy trong WSL Ubuntu** thông qua **Windows làm proxy**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETWORKING ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘

Android Phone          Windows Host              WSL Ubuntu
─────────────          ────────────              ──────────
                                                  
Flutter App    ←→    Port Proxy    ←→    FastAPI Backend
(Mobile)            (Windows)              (Python)
                                           
Port: 8000          Port: 8000            Port: 8000
IP: 192.168.1.x     IP: 192.168.1.12      IP: 172.22.105.141
                    (WiFi LAN)            (WSL internal)
```

---

## 🔍 Chi tiết Từng Layer

### Layer 1: Backend trong WSL Ubuntu

```bash
# Backend chạy trong WSL
cd /home/tran_thien/interior_project/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Listening on:
# - 0.0.0.0:8000 (accept connections from anywhere)
# - WSL IP: 172.22.105.141:8000
```

**Giải thích:**
- `--host 0.0.0.0`: Accept connections từ mọi IP (không chỉ localhost)
- `--port 8000`: Listen trên port 8000
- WSL có IP riêng trong virtual network: `172.22.105.141`

**Kiểm tra:**
```bash
# Trong WSL
curl http://localhost:8000/api/v1/health
# → Should return {"status": "ok"}

# Kiểm tra IP của WSL
ip addr show eth0
# → inet 172.22.105.141/20
```

---

### Layer 2: Windows Port Forwarding

**Vấn đề:**
- WSL có IP riêng (`172.22.105.141`) trong virtual network
- Điện thoại Android không thể truy cập trực tiếp vào WSL IP
- Cần Windows làm "cầu nối" (proxy)

**Giải pháp: Port Proxy**

```powershell
# Script: setup_port_forward.ps1
# Chạy với quyền Administrator

# Forward port 8000 từ Windows → WSL
netsh interface portproxy add v4tov4 `
  listenport=8000 `
  listenaddress=0.0.0.0 `
  connectport=8000 `
  connectaddress=172.22.105.141

# Mở firewall
netsh advfirewall firewall add rule `
  name="WSL Backend Port 8000" `
  dir=in `
  action=allow `
  protocol=TCP `
  localport=8000
```

**Giải thích:**
- `listenaddress=0.0.0.0`: Windows listen trên tất cả network interfaces
- `listenport=8000`: Windows listen trên port 8000
- `connectaddress=172.22.105.141`: Forward tới WSL IP
- `connectport=8000`: Forward tới port 8000 trong WSL

**Kết quả:**
```
Request đến Windows:8000 → Tự động forward → WSL:8000
```

**Kiểm tra:**
```powershell
# Xem port forwarding rules
netsh interface portproxy show v4tov4

# Output:
# Listen on ipv4:             Connect to ipv4:
# Address         Port        Address         Port
# --------------- ----------  --------------- ----------
# 0.0.0.0         8000        172.22.105.141  8000
```

---

### Layer 3: Flutter App trên Android

#### Option 1: USB Connection (Hiện tại KHÔNG dùng)

```dart
// config.dart
static const String _backendUrl = "http://localhost:8000";

// Setup ADB reverse
// adb reverse tcp:8000 tcp:8000
```

**Cách hoạt động:**
```
Android:localhost:8000 → ADB reverse → Windows:8000 → Port proxy → WSL:8000
```

**Ưu điểm:**
- Dùng `localhost` đơn giản
- Không cần biết IP của Windows

**Nhược điểm:**
- Phải cắm USB
- Phải chạy `adb reverse` mỗi lần
- Không test được trên nhiều thiết bị

---

#### Option 2: WiFi Connection (Đang dùng)

```dart
// config.dart
static const String _backendUrl = "http://192.168.1.12:8000";
```

**Cách hoạt động:**
```
Android (192.168.1.x) → WiFi LAN → Windows (192.168.1.12:8000) → Port proxy → WSL:8000
```

**Yêu cầu:**
1. Điện thoại và laptop cùng WiFi
2. Biết IP của Windows: `192.168.1.12`
3. Port forwarding đã setup

**Ưu điểm:**
- Không cần USB
- Test được trên nhiều thiết bị
- Giống production (mobile app gọi API qua internet)

**Nhược điểm:**
- Phải update IP nếu Windows đổi IP
- Phải cùng WiFi network

---

## 🔧 Setup Chi tiết

### Bước 1: Lấy IP của WSL

```bash
# Trong WSL Ubuntu
ip addr show eth0 | grep "inet "

# Output:
# inet 172.22.105.141/20 brd 172.22.111.255 scope global eth0
```

**Lưu IP:** `172.22.105.141`

---

### Bước 2: Lấy IP của Windows

```powershell
# Trong PowerShell
ipconfig

# Tìm "Wireless LAN adapter Wi-Fi:"
# IPv4 Address. . . . . . . . . . . : 192.168.1.12
```

**Lưu IP:** `192.168.1.12`

---

### Bước 3: Setup Port Forwarding

```powershell
# Chạy PowerShell với quyền Administrator
# Right-click PowerShell → Run as Administrator

# Chạy script
cd D:\interior_ai\interior_project
.\setup_port_forward.ps1

# Hoặc manual:
netsh interface portproxy add v4tov4 `
  listenport=8000 `
  listenaddress=0.0.0.0 `
  connectport=8000 `
  connectaddress=172.22.105.141

netsh advfirewall firewall add rule `
  name="WSL Backend Port 8000" `
  dir=in action=allow protocol=TCP localport=8000
```

---

### Bước 4: Start Backend trong WSL

```bash
# Trong WSL
cd /home/tran_thien/interior_project/backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Output:
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Bước 5: Test từ Windows

```powershell
# Test từ Windows
curl http://localhost:8000/api/v1/health

# Hoặc mở browser:
# http://localhost:8000/api/v1/health
# http://192.168.1.12:8000/api/v1/health
```

**Kết quả mong đợi:**
```json
{"status": "healthy", "message": "Backend is running"}
```

---

### Bước 6: Update Flutter Config

```dart
// frontend/lib/config.dart

class AppConfig {
  // Option 1: USB với ADB reverse
  // static const String _backendUrl = "http://localhost:8000";
  
  // Option 2: WiFi (đang dùng)
  static const String _backendUrl = "http://192.168.1.12:8000";
  
  static String get baseUrl => _backendUrl;
}
```

---

### Bước 7: Test từ Android

```bash
# Build và chạy Flutter app
cd D:\interior_ai\frontend
flutter run

# Hoặc test API từ Android browser:
# http://192.168.1.12:8000/api/v1/health
```

---

## 🐛 Troubleshooting

### Problem 1: "Connection refused" từ Android

**Nguyên nhân:**
- Port forwarding chưa setup
- Firewall block port 8000
- Backend chưa chạy

**Giải pháp:**
```powershell
# 1. Kiểm tra port forwarding
netsh interface portproxy show v4tov4

# 2. Kiểm tra firewall
netsh advfirewall firewall show rule name="WSL Backend Port 8000"

# 3. Test từ Windows trước
curl http://localhost:8000/api/v1/health
```

---

### Problem 2: "Timeout" khi gọi API

**Nguyên nhân:**
- Backend đang xử lý lâu (SAM, Stable Diffusion)
- Timeout setting quá ngắn

**Giải pháp:**
```dart
// config.dart
static const Duration receiveTimeout = Duration(minutes: 26);
```

---

### Problem 3: WSL IP thay đổi

**Nguyên nhân:**
- WSL restart → IP mới
- Windows restart → WSL IP reset

**Giải pháp:**
```bash
# 1. Lấy IP mới
ip addr show eth0 | grep "inet "

# 2. Update port forwarding
# Chạy lại setup_port_forward.ps1 với IP mới
```

**Tự động hóa:**
```powershell
# Script tự động lấy WSL IP
$wslIP = (wsl hostname -I).Trim()
Write-Host "WSL IP: $wslIP"

netsh interface portproxy add v4tov4 `
  listenport=8000 `
  listenaddress=0.0.0.0 `
  connectport=8000 `
  connectaddress=$wslIP
```

---

### Problem 4: Windows IP thay đổi

**Nguyên nhân:**
- Đổi WiFi network
- DHCP assign IP mới

**Giải pháp:**
```powershell
# 1. Lấy IP mới
ipconfig | findstr "IPv4"

# 2. Update Flutter config
# frontend/lib/config.dart
static const String _backendUrl = "http://NEW_IP:8000";

# 3. Rebuild app
flutter run
```

**Tự động hóa:**
```dart
// config.dart - Dynamic IP (advanced)
static String get baseUrl {
  if (kDebugMode) {
    // Development: Tự động detect
    return "http://${_getWindowsIP()}:8000";
  } else {
    // Production: Hardcoded
    return "http://api.yourapp.com";
  }
}
```

---

## 📊 Network Flow Diagram

### Request Flow (Chi tiết)

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE REQUEST FLOW                        │
└─────────────────────────────────────────────────────────────────┘

1. User clicks "Upload Image" in Flutter app
   │
   ├─> Flutter: ApiService.uploadImage()
   │   POST http://192.168.1.12:8000/api/v1/segmentation/upload
   │
   ├─> Android WiFi: Send HTTP request to 192.168.1.12:8000
   │   (Through WiFi router)
   │
   ├─> Windows: Receive on port 8000
   │   netsh portproxy: Forward to 172.22.105.141:8000
   │
   ├─> WSL: FastAPI receives request
   │   @router.post("/upload")
   │   async def upload_image(file: UploadFile)
   │
   ├─> WSL: Process image
   │   - Save to data/inputs/
   │   - Generate image_id
   │   - Get dimensions
   │
   ├─> WSL: Return response
   │   {"image_id": "abc-123", "image_shape": {...}}
   │
   ├─> Windows: Forward response back
   │
   ├─> Android: Receive response
   │
   └─> Flutter: Update UI with image_id
```

---

### Response Flow (Chi tiết)

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE FLOW                                │
└─────────────────────────────────────────────────────────────────┘

WSL Backend (172.22.105.141:8000)
   │
   ├─> FastAPI returns JSON response
   │   Content-Type: application/json
   │   Body: {"image_id": "abc-123", ...}
   │
   ├─> Response goes through WSL network stack
   │   172.22.105.141:8000 → Windows
   │
   ├─> Windows Port Proxy forwards response
   │   172.22.105.141:8000 → 192.168.1.12:8000
   │
   ├─> Windows sends response to Android
   │   192.168.1.12:8000 → 192.168.1.x:random_port
   │
   └─> Flutter receives and parses JSON
       final result = json.decode(response.body);
```

---

## 🔐 Security Considerations

### 1. Firewall Rules

```powershell
# Chỉ allow port 8000 từ local network
netsh advfirewall firewall add rule `
  name="WSL Backend Port 8000" `
  dir=in `
  action=allow `
  protocol=TCP `
  localport=8000 `
  remoteip=192.168.1.0/24  # Chỉ local network
```

### 2. CORS Configuration

```python
# backend/app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://192.168.1.12:8000",  # Windows
        "http://localhost:8000",      # Local testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3. Production Deployment

```dart
// config.dart
class AppConfig {
  static String get baseUrl {
    if (kReleaseMode) {
      // Production: HTTPS với domain
      return "https://api.yourapp.com";
    } else {
      // Development: Local IP
      return "http://192.168.1.12:8000";
    }
  }
}
```

---

## 📝 Checklist Setup

### Initial Setup (Làm 1 lần)
- [ ] Lấy WSL IP: `ip addr show eth0`
- [ ] Lấy Windows IP: `ipconfig`
- [ ] Chạy `setup_port_forward.ps1` (Administrator)
- [ ] Test từ Windows: `curl http://localhost:8000/api/v1/health`
- [ ] Update Flutter config với Windows IP
- [ ] Test từ Android browser

### Mỗi lần Restart
- [ ] Start backend trong WSL: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- [ ] Kiểm tra port forwarding: `netsh interface portproxy show v4tov4`
- [ ] Test từ Windows: `curl http://localhost:8000/api/v1/health`

### Khi Đổi Network
- [ ] Lấy Windows IP mới: `ipconfig`
- [ ] Update Flutter config
- [ ] Rebuild Flutter app: `flutter run`

---

## 🎓 Trong Báo cáo

### Phần System Architecture

> **3.2 Network Architecture**
> 
> Hệ thống sử dụng kiến trúc client-server với 3 layers:
> 
> 1. **Client Layer (Flutter Mobile App)**
>    - Chạy trên Android device
>    - Giao tiếp qua HTTP REST API
>    - IP: 192.168.1.x (WiFi LAN)
> 
> 2. **Proxy Layer (Windows Host)**
>    - Windows Port Proxy forward requests
>    - IP: 192.168.1.12 (WiFi LAN)
>    - Port: 8000
>    - Firewall rules cho local network access
> 
> 3. **Server Layer (WSL Ubuntu)**
>    - FastAPI backend với Python
>    - IP: 172.22.105.141 (WSL virtual network)
>    - Port: 8000
>    - Host: 0.0.0.0 (accept external connections)
> 
> **Network Flow:**
> ```
> Android (192.168.1.x) → WiFi → Windows (192.168.1.12) 
>   → Port Proxy → WSL (172.22.105.141) → FastAPI
> ```
> 
> **Lý do sử dụng WSL:**
> - GPU support cho CUDA (SAM, Stable Diffusion)
> - Linux environment cho Python ML libraries
> - Dễ deploy lên Linux server sau này

---

## 🔗 References

- [WSL Networking](https://docs.microsoft.com/en-us/windows/wsl/networking)
- [Windows Port Proxy](https://docs.microsoft.com/en-us/windows-server/networking/technologies/netsh/netsh-interface-portproxy)
- [Flutter HTTP](https://docs.flutter.dev/cookbook/networking/fetch-data)

---

**Tóm tắt:** Flutter (Android) → WiFi → Windows Port Proxy → WSL Backend
