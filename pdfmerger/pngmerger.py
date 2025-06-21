import os
from PIL import Image
import re

TOP_CROP = 250
BOTTOM_CROP = 300
TARGET_HEIGHT = 1080


def crop_image(image, top_crop=TOP_CROP, bottom_crop=BOTTOM_CROP):
    """
    이미지의 위쪽과 아래쪽을 잘라냅니다.
    
    Args:
        image: PIL Image 객체
        top_crop (int): 위쪽에서 잘라낼 픽셀 수
        bottom_crop (int): 아래쪽에서 잘라낼 픽셀 수
    
    Returns:
        PIL Image: 잘라진 이미지
    """
    width, height = image.size
    # (left, top, right, bottom) 형식으로 크롭 영역 지정
    crop_box = (0, top_crop, width, height - bottom_crop)
    return image.crop(crop_box)

def resize_to_target_height(image, target_height=TARGET_HEIGHT):
    """
    이미지의 세로 해상도를 목표값으로 조정하고 가로 비율을 유지합니다.
    
    Args:
        image: PIL Image 객체
        target_height (int): 목표 세로 해상도
    
    Returns:
        PIL Image: 리사이즈된 이미지
    """
    width, height = image.size
    
    # 비율 계산
    scale_ratio = target_height / height
    new_width = int(width * scale_ratio)
    
    # 고품질 리사이즈 (LANCZOS 사용)
    resized_image = image.resize((new_width, target_height), Image.Resampling.LANCZOS)
    
    return resized_image

def combine_sheet_music_pages(input_folder, output_folder):
    """
    악보 PNG 파일들을 두 페이지씩 합쳐서 새로운 이미지 파일로 저장합니다.
    각 페이지는 위쪽 250픽셀, 아래쪽 300픽셀을 잘라내고, 최종 세로 해상도를 1080으로 조정합니다.
    
    Args:
        input_folder (str): 원본 PNG 파일들이 있는 폴더 경로
        output_folder (str): 합쳐진 이미지를 저장할 폴더 경로
    """
    
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # PNG 파일들을 찾아서 정렬
    png_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith('.png') and 'la gazza ladra' in file.lower():
            png_files.append(file)
    
    # 파일명에서 페이지 번호 추출하여 정렬
    def extract_page_number(filename):
        # 파일명에서 숫자 추출 (예: "이지_01.png" -> 1)
        match = re.search(r'(\d+)\.png$', filename)
        if match:
            return int(match.group(1))
        return 0
    
    png_files.sort(key=extract_page_number)
    
    print(f"총 {len(png_files)}개의 파일을 발견했습니다.")
    
    # 두 페이지씩 짝지어서 합치기
    for i in range(0, len(png_files), 2):
        try:
            # 첫 번째 페이지 로드 및 크롭
            img1_path = os.path.join(input_folder, png_files[i])
            img1_original = Image.open(img1_path)
            img1 = crop_image(img1_original, top_crop=TOP_CROP, bottom_crop=BOTTOM_CROP)
            img1_original.close()  # 원본 이미지 메모리 해제
            
            # 두 번째 페이지가 있으면 로드 및 크롭, 없으면 첫 번째 페이지만 사용
            if i + 1 < len(png_files):
                img2_path = os.path.join(input_folder, png_files[i + 1])
                img2_original = Image.open(img2_path)
                img2 = crop_image(img2_original, top_crop=TOP_CROP, bottom_crop=BOTTOM_CROP)
                img2_original.close()  # 원본 이미지 메모리 해제
                
                # 두 이미지의 크기 확인
                width1, height1 = img1.size
                width2, height2 = img2.size
                
                # 새로운 이미지 크기 계산 (가로로 나란히 배치)
                new_width = width1 + width2
                new_height = max(height1, height2)
                
                # 새로운 이미지 생성 (흰색 배경)
                combined_img = Image.new('RGB', (new_width, new_height), 'white')
                
                # 이미지들을 새로운 이미지에 붙여넣기
                combined_img.paste(img1, (0, 0))
                combined_img.paste(img2, (width1, 0))
                
                # 파일명 생성
                page_num = extract_page_number(png_files[i])
                output_filename = f"combined_pages_{page_num:02d}_{page_num+1:02d}.png"
                
                print(f"페이지 {page_num}, {page_num+1} 합치는 중...")
                
            else:
                # 마지막 페이지가 홀수인 경우 - 빈 페이지 추가
                width1, height1 = img1.size
                
                # 새로운 이미지 크기 계산 (가로로 나란히 배치, 빈 페이지 추가)
                new_width = width1 * 2
                new_height = height1
                
                # 새로운 이미지 생성 (흰색 배경)
                combined_img = Image.new('RGB', (new_width, new_height), 'white')
                
                # 첫 번째 페이지만 붙여넣기 (왼쪽에), 오른쪽은 빈 페이지로 남김
                combined_img.paste(img1, (0, 0))
                
                page_num = extract_page_number(png_files[i])
                output_filename = f"combined_pages_{page_num:02d}_blank.png"
                
                print(f"마지막 페이지 {page_num} + 빈 페이지 처리 중...")
            
            # 최종 이미지를 1080 높이로 리사이즈
            final_img = resize_to_target_height(combined_img, TARGET_HEIGHT)
            combined_img.close()  # 원본 합친 이미지 메모리 해제
            
            # 합쳐진 이미지 저장
            output_path = os.path.join(output_folder, output_filename)
            final_img.save(output_path, 'PNG', quality=95)
            
            final_width, final_height = final_img.size
            print(f"저장 완료: {output_filename} (최종 크기: {final_width}x{final_height})")
            
            # 메모리 정리
            img1.close()
            if i + 1 < len(png_files):
                img2.close()
            final_img.close()
            
        except Exception as e:
            print(f"오류 발생 (파일 {i//2 + 1}): {e}")
            continue
    
    print("모든 파일 처리가 완료되었습니다!")

if __name__ == "__main__":
    # 사용 예시
    input_folder = r"D:\Dropbox\미련\미련2025\악보\La Gazza ladra"  # 원본 PNG 파일들이 있는 폴더
    output_folder = r"D:\Dropbox\미련\미련2025\악보\La Gazza ladra\combined_horizontal"  # 가로로 합친 파일들을 저장할 폴더
    
    print("=== 4961x7016 악보 페이지 합치기 (위쪽 250px, 아래쪽 300px 크롭) ===")
    print("=== 가로로 페이지 합치기 + 1080p 리사이즈 ===")
    combine_sheet_music_pages(input_folder, output_folder)
    
    print("\n작업이 모두 완료되었습니다!")
    print(f"원본 크기: 4961x7016 → 크롭 후: 4961x6466")
    print(f"합친 파일 크기: 9922x6466 → 최종: 약 1664x1080")
    print(f"결과 파일: {output_folder} 폴더")
