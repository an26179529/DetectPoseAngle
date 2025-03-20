import cv2
import mediapipe as mp
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import sys

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化姿勢偵測器
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5)
        
    def analyze_pose(self, frame, body_side="both"):
        # 轉換為RGB (MediaPipe需要)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        # 建立框架用於繪圖
        output_frame = frame.copy()
        
        if results.pose_landmarks:
            # 獲取尺寸
            h, w = output_frame.shape[:2]
            
            # 繪製骨架
            self.mp_drawing.draw_landmarks(
                output_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            
            # 獲取關鍵點
            landmarks = results.pose_landmarks.landmark
            
            # 右半身關鍵點
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            # 左半身關鍵點
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # 建立髖關節中點
            hip_mid_x = (left_hip.x + right_hip.x) / 2
            hip_mid_y = (left_hip.y + right_hip.y) / 2
            hip_mid_z = (left_hip.z + right_hip.z) / 2
            
            class VirtualPoint:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z
            
            hip_midpoint = VirtualPoint(hip_mid_x, hip_mid_y, hip_mid_z)
            
            # 計算胯下角度 (左膝蓋、髖關節中點、右膝蓋)
            hip_width_angle = self.calculate_angle(
                left_knee, hip_midpoint, right_knee, w, h
            )
            
            # 計算右側身體-腿部角度 (肩膀、髖部、腳踝)
            right_body_leg_angle = self.calculate_angle(
                right_shoulder, right_hip, right_ankle, w, h
            )
            
            # 計算左側身體-腿部角度 (肩膀、髖部、腳踝)
            left_body_leg_angle = self.calculate_angle(
                left_shoulder, left_hip, left_ankle, w, h
            )
            
            # 繪製角度
            self.draw_angle_lines(
                output_frame, 
                hip_width_angle, right_body_leg_angle, left_body_leg_angle,
                hip_midpoint, left_knee, right_knee,
                right_shoulder, right_hip, right_ankle,
                left_shoulder, left_hip, left_ankle,
                w, h, body_side
            )
            
            return output_frame, hip_width_angle, right_body_leg_angle, left_body_leg_angle
        
        # 若未偵測到骨架，回傳原始框架
        return output_frame, None, None, None
    
    def calculate_angle(self, p1, p2, p3, width, height):
        """計算三點間角度"""
        try:
            # 計算向量
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            # 計算向量長度
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm == 0 or v2_norm == 0:
                return 0
                
            # 計算點積
            dot_product = np.dot(v1, v2)
            
            # 計算角度 (弧度)
            angle_rad = np.arccos(np.clip(dot_product / (v1_norm * v2_norm), -1.0, 1.0))
            
            # 轉換為角度
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
        except:
            return 0
    
    def draw_angle_lines(self, frame, hip_width_angle, right_body_leg_angle, left_body_leg_angle,
                       hip_midpoint, left_knee, right_knee,
                       right_shoulder, right_hip, right_ankle,
                       left_shoulder, left_hip, left_ankle,
                       width, height, body_side="both"):
        """繪製角度線條"""
        # 轉換為像素座標
        def to_pixel(landmark):
            return (int(landmark.x * width), int(landmark.y * height))
        
        # 獲取所有需要的像素座標
        hip_mid_px = to_pixel(hip_midpoint)
        left_knee_px = to_pixel(left_knee)
        right_knee_px = to_pixel(right_knee)
        
        right_shoulder_px = to_pixel(right_shoulder)
        right_hip_px = to_pixel(right_hip)
        right_ankle_px = to_pixel(right_ankle)
        
        left_shoulder_px = to_pixel(left_shoulder)
        left_hip_px = to_pixel(left_hip)
        left_ankle_px = to_pixel(left_ankle)
        
        # 標記髖關節中點
        cv2.circle(frame, hip_mid_px, 8, (255, 0, 255), -1)
        
        # 根據選擇的身體側繪製
        if body_side in ["both", "left", "right"]:
            # 繪製胯下角度 (紅色)
            cv2.line(frame, hip_mid_px, left_knee_px, (0, 0, 255), 3)
            cv2.line(frame, hip_mid_px, right_knee_px, (0, 0, 255), 3)
            
            # 加入胯下角度文字
            cv2.putText(frame, f"Hip Width: {hip_width_angle:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 繪製左側
        if body_side in ["both", "left"]:
            # 繪製左側身體-腿部角度 (綠色)
            cv2.line(frame, left_shoulder_px, left_hip_px, (0, 255, 0), 3)
            cv2.line(frame, left_hip_px, left_ankle_px, (0, 255, 0), 3)
            
            # 加入左側角度文字
            cv2.putText(frame, f"L Body-Leg: {left_body_leg_angle:.1f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 繪製右側
        if body_side in ["both", "right"]:
            # 繪製右側身體-腿部角度 (綠色)
            cv2.line(frame, right_shoulder_px, right_hip_px, (0, 255, 0), 3)
            cv2.line(frame, right_hip_px, right_ankle_px, (0, 255, 0), 3)
            
            # 加入右側角度文字
            cv2.putText(frame, f"R Body-Leg: {right_body_leg_angle:.1f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

class PoseAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("姿勢分析系統")
        self.root.geometry("1000x700")
        
        self.analyzer = PoseAnalyzer()
        self.input_path = None
        self.output_path = None
        self.is_photo = True  # 預設為照片模式
        self.body_side = "both"  # 預設為顯示兩側
        self.is_processing = False
        self.preview_image = None
        self.cap = None  # 用於影片捕捉
        
        self.setup_ui()
    
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左側控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 輸入類型選項
        type_frame = ttk.Frame(control_frame)
        type_frame.pack(fill=tk.X, pady=5)
        
        self.input_type = tk.StringVar(value="photo")
        ttk.Label(type_frame, text="輸入類型:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="照片", value="photo", 
                      variable=self.input_type, command=self.on_type_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="影片", value="video", 
                      variable=self.input_type, command=self.on_type_change).pack(side=tk.LEFT, padx=5)
        
        # 身體側選項
        side_frame = ttk.Frame(control_frame)
        side_frame.pack(fill=tk.X, pady=5)
        
        self.body_side_var = tk.StringVar(value="both")
        ttk.Label(side_frame, text="顯示側面:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(side_frame, text="左側", value="left", 
                      variable=self.body_side_var).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(side_frame, text="右側", value="right", 
                      variable=self.body_side_var).pack(side=tk.LEFT, padx=5)
        
        # 文件選擇
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="檔案:").pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="瀏覽", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        
        self.file_label = ttk.Label(control_frame, text="未選擇檔案")
        self.file_label.pack(fill=tk.X, pady=5)
        
        # 輸出路徑
        output_frame = ttk.Frame(control_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="輸出:").pack(side=tk.LEFT, padx=5)
        ttk.Button(output_frame, text="瀏覽", command=self.browse_output).pack(side=tk.LEFT, padx=5)
        
        self.output_label = ttk.Label(control_frame, text="未選擇輸出路徑")
        self.output_label.pack(fill=tk.X, pady=5)
        
        # 控制按鈕
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=15)
        
        self.analyze_btn = ttk.Button(button_frame, text="分析", command=self.start_analysis)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="停止", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 狀態欄
        status_frame = ttk.LabelFrame(control_frame, text="狀態")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="就緒")
        ttk.Label(status_frame, textvariable=self.status_var).pack(pady=5)
        
        # 進度條
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # 預覽區域
        preview_frame = ttk.LabelFrame(main_frame, text="預覽")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(preview_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
    def on_type_change(self):
        self.is_photo = self.input_type.get() == "photo"
        self.input_path = None
        self.file_label.config(text="未選擇檔案")
        self.canvas.delete("all")
        self.status_var.set(f"已切換到{'照片' if self.is_photo else '影片'}模式")
    
    def browse_file(self):
        if self.is_photo:
            file_types = [("圖片檔", "*.jpg *.jpeg *.png"), ("所有檔案", "*.*")]
        else:
            file_types = [("影片檔", "*.mp4 *.avi *.mov"), ("所有檔案", "*.*")]
        
        file_path = filedialog.askopenfilename(
            title=f"選擇{'照片' if self.is_photo else '影片'}",
            filetypes=file_types
        )
        
        if file_path:
            self.input_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.show_preview()
    
    def browse_output(self):
        dir_path = filedialog.askdirectory(title="選擇輸出資料夾")
        if dir_path:
            self.output_path = dir_path
            self.output_label.config(text=os.path.basename(dir_path))
    
    def show_preview(self):
        if not self.input_path:
            return
        
        if self.is_photo:
            # 顯示照片預覽
            try:
                img = cv2.imread(self.input_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.resize_for_display(img)
                
                # 轉換成PhotoImage
                img_pil = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                # 在畫布上顯示
                self.canvas.config(width=img_tk.width(), height=img_tk.height())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.preview_image = img_tk  # 保留引用
            except Exception as e:
                messagebox.showerror("錯誤", f"無法載入照片: {e}")
        else:
            # 顯示影片第一幀
            try:
                cap = cv2.VideoCapture(self.input_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = self.resize_for_display(frame)
                    
                    # 轉換成PhotoImage
                    img_pil = Image.fromarray(frame)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    
                    # 在畫布上顯示
                    self.canvas.config(width=img_tk.width(), height=img_tk.height())
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                    self.preview_image = img_tk  # 保留引用
                else:
                    messagebox.showerror("錯誤", "無法讀取影片")
            except Exception as e:
                messagebox.showerror("錯誤", f"無法載入影片: {e}")
    
    def resize_for_display(self, image, max_width=800, max_height=600):
        h, w = image.shape[:2]
        
        # 計算縮放比例
        aspect = w / h
        
        # 縮放到最大尺寸內，保持畫面比例
        if w > max_width:
            w = max_width
            h = int(w / aspect)
        
        if h > max_height:
            h = max_height
            w = int(h * aspect)
        
        return cv2.resize(image, (w, h))
    
    def start_analysis(self):
        if not self.input_path:
            messagebox.showwarning("警告", "請先選擇檔案")
            return
        
        if not self.output_path:
            messagebox.showwarning("警告", "請選擇輸出資料夾")
            return
        
        self.is_processing = True
        self.analyze_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        if self.is_photo:
            self.process_photo()
        else:
            self.process_video()
    
    def stop_analysis(self):
        self.is_processing = False
        self.status_var.set("已停止")
        self.analyze_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        # 釋放影片捕捉器（若存在）
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def process_photo(self):
        try:
            # 讀取圖片
            img = cv2.imread(self.input_path)
            if img is None:
                raise Exception("無法讀取照片")
            
            # 處理圖片 - 傳入側面選擇
            self.status_var.set("分析中...")
            processed_img, hip_width_angle, right_body_leg_angle, left_body_leg_angle = self.analyzer.analyze_pose(
                img, self.body_side_var.get()
            )
            
            # 生成輸出檔名
            filename = os.path.basename(self.input_path)
            name, ext = os.path.splitext(filename)
            output_file = os.path.join(self.output_path, f"{name}_analyzed{ext}")
            
            # 儲存處理後的圖片
            cv2.imwrite(output_file, processed_img)
            
            # 顯示處理後的圖片
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            processed_img = self.resize_for_display(processed_img)
            
            img_pil = Image.fromarray(processed_img)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.canvas.config(width=img_tk.width(), height=img_tk.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.preview_image = img_tk  # 保留引用
            
            self.status_var.set(f"分析完成，已儲存至 {output_file}")
            self.analyze_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("錯誤", f"處理照片時發生錯誤: {e}")
            self.status_var.set("處理失敗")
            self.analyze_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def process_video(self):
        try:
            self.cap = cv2.VideoCapture(self.input_path)
            if not self.cap.isOpened():
                raise Exception("無法開啟影片")
            
            # 獲取影片屬性
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 建立輸出影片寫入器
            filename = os.path.basename(self.input_path)
            name, ext = os.path.splitext(filename)
            output_file = os.path.join(self.output_path, f"{name}_analyzed{ext}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            # 處理每一幀
            frame_count = 0
            while self.is_processing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 處理幀 - 傳入側面選擇
                processed_frame, hip_width_angle, right_body_leg_angle, left_body_leg_angle = self.analyzer.analyze_pose(
                    frame, self.body_side_var.get()
                )
                
                # 寫入輸出影片
                out.write(processed_frame)
                
                # 更新進度
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"處理中... {progress:.1f}%")
                
                # 顯示當前幀（每5幀更新一次以減少負載）
                if frame_count % 5 == 0:
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    display_frame = self.resize_for_display(display_frame)
                    
                    img_pil = Image.fromarray(display_frame)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    
                    self.canvas.config(width=img_tk.width(), height=img_tk.height())
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                    self.preview_image = img_tk  # 保留引用
                
                # 更新UI
                self.root.update()
            
            # 清理資源
            out.release()
            self.cap.release()
            self.cap = None
            
            if not self.is_processing:
                self.status_var.set("處理已中止")
            else:
                self.status_var.set(f"處理完成，已儲存至 {output_file}")
            
            self.analyze_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.is_processing = False
            
        except Exception as e:
            messagebox.showerror("錯誤", f"處理影片時發生錯誤: {e}")
            self.status_var.set("處理失敗")
            self.analyze_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
            if self.cap is not None:
                self.cap.release()
                self.cap = Nonevar.set(progress)
                self.status_var.set(f"處理中... {progress:.1f}%")
                
                # 顯示當前幀（每5幀更新一次以減少負載）
                if frame_count % 5 == 0:
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    display_frame = self.resize_for_display(display_frame)
                    
                    img_pil = Image.fromarray(display_frame)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    
                    self.canvas.config(width=img_tk.width(), height=img_tk.height())
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                    self.preview_image = img_tk  # 保留引用
                
                # 更新UI
                self.root.update()
            
            # 清理資源
            out.release()
            self.cap.release()
            self.cap = None
            
            if not self.is_processing:
                self.status_var.set("處理已中止")
            else:
                self.status_var.set(f"處理完成，已儲存至 {output_file}")
            
            self.analyze_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.is_processing = False
            
        except Exception as e:
            messagebox.showerror("錯誤", f"處理影片時發生錯誤: {e}")
            self.status_var.set("處理失敗")
            self.analyze_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
            if self.cap is not None:
                self.cap.release()
                self.cap = None

def main():
    try:
        # 設定標準輸出編碼
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
    
    root = tk.Tk()
    app = PoseAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()