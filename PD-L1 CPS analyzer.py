import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2

# scikit-image Library import
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.morphology import disk, opening
from skimage.measure import find_contours
from skimage.filters import threshold_otsu

# Color threshold setting (initial preset)
# Scikit-image usese thresholds of H(0-1), S(0-1), V(0-1)
# Set inital threshold of H and S as Low=0, High=1.00
DEFAULT_LOWER_DAB = [0.0, 0.0, 0.2]
DEFAULT_UPPER_DAB = [1.0, 1.0, 1.0]
DEFAULT_LOWER_NUCLEUS = [0.0, 0.0, 0.2]
DEFAULT_UPPER_NUCLEUS = [1.0, 1.0, 1.0]

# Nucleus detection threshold setting
DEFAULT_MIN_NUCLEUS_AREA = 50 
DEFAULT_MAX_NUCLEUS_AREA = 200
DEFAULT_MIN_DAB_AREA = 50 
DEFAULT_MAX_DAB_AREA = 200

class PDL1AnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PD-L1 IHC Analyzer (Scikit-image)")
        self.image_path = None
        self.result_image = None
        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=(0, 20), fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.select_button = tk.Button(left_frame, text="이미지 선택", command=self.select_image)
        self.select_button.pack(pady=5)
        self.analyze_button = tk.Button(left_frame, text="분석 시작", command=self.analyze_image, state=tk.DISABLED)
        self.analyze_button.pack(pady=5)

        # Options for analysis methods
        self.analysis_mode = tk.StringVar(value="bounding_box")
        mode_frame = tk.LabelFrame(left_frame, text="분석 방식 선택", padx=10, pady=5)
        mode_frame.pack(pady=5)
        tk.Radiobutton(mode_frame, text="바운딩 박스", variable=self.analysis_mode, value="bounding_box").pack(anchor="w")
        tk.Radiobutton(mode_frame, text="오버레이", variable=self.analysis_mode, value="overlay").pack(anchor="w")

        self.result_label = tk.Label(left_frame, text="분석 결과: 대기 중", font=("Arial", 12))
        self.result_label.pack(pady=10)

        self.image_canvas = tk.Canvas(left_frame, width=800, height=600, bg="gray")
        self.image_canvas.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.save_button = tk.Button(left_frame, text="결과 이미지 저장", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        # Scrollable frame for parameters (thresholds)
        canvas = tk.Canvas(right_frame, width=300)
        scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.auto_threshold_button = tk.Button(scrollable_frame, text="자동 임계값 설정", command=self.set_auto_thresholds)
        self.auto_threshold_button.pack(pady=5, fill=tk.X)

        # --- 파라미터 (슬라이더) 구현 ---
        
        # descriptions
        hsv_desc_frame = tk.LabelFrame(scrollable_frame, text="HSV Values Guide", padx=10, pady=10)
        hsv_desc_frame.pack(pady=10, fill=tk.X)
        tk.Label(hsv_desc_frame, text="H (색상): 0-1.0 범위. 색의 종류를 지정합니다.\n낮은 값은 붉은색 계열, 높은 값은 푸른색 계열입니다.", justify=tk.LEFT).pack(anchor="w")
        tk.Label(hsv_desc_frame, text="S (채도): 0-1.0 범위. 색의 선명도를 지정합니다.\n값이 높을수록 선명하고, 낮을수록 탁해집니다.", justify=tk.LEFT).pack(anchor="w")
        tk.Label(hsv_desc_frame, text="V (명도): 0-1.0 범위. 색의 밝기를 지정합니다.\n값이 높을수록 밝고, 낮을수록 어두워집니다.", justify=tk.LEFT).pack(anchor="w")

        # threshold for DAB
        dab_frame = tk.LabelFrame(scrollable_frame, text="DAB (PD-L1 Positive) Thresholds", padx=10, pady=10)
        dab_frame.pack(pady=10, fill=tk.X)
        self.lower_h_dab_scale = self.create_hsv_scale(dab_frame, "Lower H:", DEFAULT_LOWER_DAB[0], 0, 1.0)
        self.upper_h_dab_scale = self.create_hsv_scale(dab_frame, "Upper H:", DEFAULT_UPPER_DAB[0], 0, 1.0)
        self.lower_s_dab_scale = self.create_hsv_scale(dab_frame, "Lower S:", DEFAULT_LOWER_DAB[1], 0, 1.0)
        self.upper_s_dab_scale = self.create_hsv_scale(dab_frame, "Upper S:", DEFAULT_UPPER_DAB[1], 0, 1.0)
        self.lower_v_dab_scale = self.create_hsv_scale(dab_frame, "Lower V:", DEFAULT_LOWER_DAB[2], 0, 1.0)
        self.upper_v_dab_scale = self.create_hsv_scale(dab_frame, "Upper V:", DEFAULT_UPPER_DAB[2], 0, 1.0)
        
        # threshold for nucleus
        nucleus_frame = tk.LabelFrame(scrollable_frame, text="Nucleus Thresholds", padx=10, pady=10)
        nucleus_frame.pack(pady=10, fill=tk.X)
        self.lower_h_nucleus_scale = self.create_hsv_scale(dab_frame, "Lower H:", DEFAULT_LOWER_NUCLEUS[0], 0, 1.0)
        self.upper_h_nucleus_scale = self.create_hsv_scale(dab_frame, "Upper H:", DEFAULT_UPPER_NUCLEUS[0], 0, 1.0)
        self.lower_s_nucleus_scale = self.create_hsv_scale(dab_frame, "Lower S:", DEFAULT_LOWER_NUCLEUS[1], 0, 1.0)
        self.upper_s_nucleus_scale = self.create_hsv_scale(dab_frame, "Upper S:", DEFAULT_UPPER_NUCLEUS[1], 0, 1.0)
        self.lower_v_nucleus_scale = self.create_hsv_scale(dab_frame, "Lower V:", DEFAULT_LOWER_NUCLEUS[2], 0, 1.0)
        self.upper_v_nucleus_scale = self.create_hsv_scale(dab_frame, "Upper V:", DEFAULT_UPPER_NUCLEUS[2], 0, 1.0)

        # Nucelus detection threshold setting
        area_frame = tk.LabelFrame(scrollable_frame, text="Object Area Thresholds", padx=10, pady=10)
        area_frame.pack(pady=10, fill=tk.X)
        self.min_dab_area_scale = self.create_area_scale(area_frame, "DAB Min Area:", DEFAULT_MIN_DAB_AREA, 0, 500)
        self.max_dab_area_scale = self.create_area_scale(area_frame, "DAB Max Area:", DEFAULT_MAX_DAB_AREA, 0, 20000)
        self.min_nucleus_area_scale = self.create_area_scale(area_frame, "Nucleus Min Area:", DEFAULT_MIN_NUCLEUS_AREA, 0, 500)
        self.max_nucleus_area_scale = self.create_area_scale(area_frame, "Nucleus Max Area:", DEFAULT_MAX_NUCLEUS_AREA, 0, 20000)

    def create_hsv_scale(self, parent, label_text, default_value, from_, to):
        scale = tk.Scale(parent, from_=from_, to=to, resolution=0.01, orient=tk.HORIZONTAL, label=label_text)
        scale.set(default_value)
        scale.pack(fill=tk.X, padx=5, pady=2)
        return scale

    def create_area_scale(self, parent, label_text, default_value, from_, to):
        scale = tk.Scale(parent, from_=from_, to=to, resolution=1, orient=tk.HORIZONTAL, label=label_text)
        scale.set(default_value)
        scale.pack(fill=tk.X, padx=5, pady=2)
        return scale

    def set_auto_thresholds(self):
        if not self.image_path:
            messagebox.showinfo("정보", "먼저 이미지를 선택하세요.")
            return

        try:
            original_image = imread(self.image_path)
            hsv_image = rgb2hsv(original_image)
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패: {e}")
            return

        try:
            val_thresh = threshold_otsu(hsv_image[:, :, 2])
        except ValueError:
            messagebox.showerror("오류", "오츠 임계값을 계산할 수 없습니다. 이미지 품질이 낮거나 객체가 너무 적을 수 있습니다.")
            return

        self.lower_v_dab_scale.set(0.0)
        self.upper_v_dab_scale.set(val_thresh)
        
        self.lower_v_nucleus_scale.set(val_thresh)
        self.upper_v_nucleus_scale.set(1.0)

        messagebox.showinfo("완료", "V(명도) 임계값이 자동으로 설정되었습니다. 다른 값들을 조정하고 '분석 시작'을 눌러주세요.")

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            initialdir="/",
            title="이미지 파일을 선택하세요",
            filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*"))
        )
        if self.image_path:
            self.display_image(self.image_path)
            self.analyze_button.config(state=tk.NORMAL)

    def display_image(self, path):
        original_image = Image.open(path)
        img_width, img_height = original_image.size
        ratio = min(800 / img_width, 600 / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(400, 300, image=self.tk_image)
        self.image_canvas.image = self.tk_image

    def analyze_image(self):
        if not self.image_path:
            messagebox.showerror("오류", "먼저 이미지를 선택하세요.")
            return

        # Reading thresholds from GUI sliders
        lower_dab = [self.lower_h_dab_scale.get(), self.lower_s_dab_scale.get(), self.lower_v_dab_scale.get()]
        upper_dab = [self.upper_h_dab_scale.get(), self.upper_s_dab_scale.get(), self.upper_v_dab_scale.get()]
        lower_nucleus = [self.lower_h_nucleus_scale.get(), self.lower_s_nucleus_scale.get(), self.lower_v_nucleus_scale.get()]
        upper_nucleus = [self.upper_h_nucleus_scale.get(), self.upper_s_nucleus_scale.get(), self.upper_v_nucleus_scale.get()]
        min_nucleus_area = self.min_nucleus_area_scale.get()
        max_nucleus_area = self.max_nucleus_area_scale.get()
        min_dab_area = self.min_dab_area_scale.get()
        max_dab_area = self.max_dab_area_scale.get()

        try:
            original_image = imread(self.image_path)
            pil_image = Image.fromarray(original_image)
        except Exception:
            messagebox.showerror("오류", "이미지를 불러올 수 없습니다. 파일 형식을 확인해주세요.")
            return

        hsv_image = rgb2hsv(original_image)
        
        dab_mask = ( (hsv_image[:, :, 0] >= lower_dab[0]) & (hsv_image[:, :, 0] <= upper_dab[0]) &
                     (hsv_image[:, :, 1] >= lower_dab[1]) & (hsv_image[:, :, 1] <= upper_dab[1]) &
                     (hsv_image[:, :, 2] >= lower_dab[2]) & (hsv_image[:, :, 2] <= upper_dab[2]) )
        
        nucleus_mask = ( (hsv_image[:, :, 0] >= lower_nucleus[0]) & (hsv_image[:, :, 0] <= upper_nucleus[0]) &
                        (hsv_image[:, :, 1] >= lower_nucleus[1]) & (hsv_image[:, :, 1] <= upper_nucleus[1]) &
                        (hsv_image[:, :, 2] >= lower_nucleus[2]) & (hsv_image[:, :, 2] <= upper_nucleus[2]) )

        kernel = disk(3) 
        dab_mask = opening(dab_mask, kernel)
        nucleus_mask = opening(nucleus_mask, kernel)

        dab_contours = find_contours(dab_mask, 0.8)
        nucleus_contours = find_contours(nucleus_mask, 0.8)

        pd_l1_pos_count = 0
        total_tumor_cell_count = 0
        
        # Image generation
        mode = self.analysis_mode.get()

        if mode == "overlay":
            # Overlay mode
            overlay = np.copy(original_image)
            overlay[dab_mask] = [255, 0, 0] # 초록색
            overlay[nucleus_mask] = [0, 255, 0] # 빨간색
            alpha = 0.5
            blended_image = (original_image * (1 - alpha) + overlay * alpha).astype(np.uint8)
            self.result_image = blended_image
            
            # counts
            for contour in dab_contours:
                # 최대/최소 면적 조건 추가
                if min_dab_area <= len(contour) <= max_dab_area:
                    pd_l1_pos_count += 1
            
            for contour in nucleus_contours:
                # 최대/최소 면적 조건 추가
                if min_nucleus_area <= len(contour) <= max_nucleus_area:
                    total_tumor_cell_count += 1

        elif mode == "bounding_box":
            # Bounding box mode
            pil_image_for_draw = Image.fromarray(original_image)
            draw = ImageDraw.Draw(pil_image_for_draw)
            
            # Bounding box for DAB
            for contour in dab_contours:
                # 최대/최소 면적 조건 추가
                if min_dab_area <= len(contour) <= max_dab_area:
                    min_row, min_col = np.min(contour, axis=0)
                    max_row, max_col = np.max(contour, axis=0)
                    draw.rectangle([(min_col, min_row), (max_col, max_row)], outline="red", width=2)
                    pd_l1_pos_count += 1

            # Bounding box for nucleus
            for contour in nucleus_contours:
                # 최대/최소 면적 조건 추가
                if min_nucleus_area <= len(contour) <= max_nucleus_area:
                    min_row, min_col = np.min(contour, axis=0)
                    max_row, max_col = np.max(contour, axis=0)
                    draw.rectangle([(min_col, min_row), (max_col, max_row)], outline="blue", width=2)
                    total_tumor_cell_count += 1

            self.result_image = np.array(pil_image_for_draw)
        
        # CPS calculation
        cps = (pd_l1_pos_count / (pd_l1_pos_count + total_tumor_cell_count)) * 100 if pd_l1_pos_count + total_tumor_cell_count > 0 else 0

        self.result_label.config(text=f"분석 완료!\n\nPD-L1 양성 세포 수: {pd_l1_pos_count}\n총 종양 세포 수: {total_tumor_cell_count}\nCPS: {cps:.2f}")

        # Image display
        pil_final_image = Image.fromarray(self.result_image)
        img_width, img_height = pil_final_image.size
        ratio = min(800 / img_width, 600 / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        resized_image = pil_final_image.resize((new_width, new_height), Image.LANCZOS)
        
        self.image_canvas.delete("all")
        self.tk_result_image = ImageTk.PhotoImage(resized_image)
        self.image_canvas.create_image(400, 300, image=self.tk_result_image)
        self.image_canvas.image = self.tk_result_image
        
        self.save_button.config(state=tk.NORMAL)
    
    def save_image(self):
        if self.result_image is not None:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*"))
            )
            if save_path:
                Image.fromarray(self.result_image).save(save_path)
                messagebox.showinfo("저장 완료", f"이미지가 성공적으로 저장되었습니다:\n{save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PDL1AnalyzerGUI(root)
    root.mainloop()
