import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import numpy as np


class GIFProcessor:
    def __init__(self):
        self.upload = widgets.FileUpload(
            accept='.gif',
            multiple=False,
            description='上传GIF'
        )

        self.width_slider = widgets.IntSlider(
            value=200,
            min=50,
            max=1000,
            step=10,
            description='宽度:',
            disabled=False
        )

        self.height_slider = widgets.IntSlider(
            value=200,
            min=50,
            max=1000,
            step=10,
            description='高度:',
            disabled=False
        )

        self.keep_aspect = widgets.Checkbox(
            value=True,
            description='保持宽高比',
            disabled=False
        )

        self.watermark_text = widgets.Text(
            value='',
            placeholder='输入水印文字',
            description='水印文字:',
            disabled=False
        )

        self.watermark_size = widgets.IntSlider(
            value=20,
            min=10,
            max=50,
            step=1,
            description='水印大小:',
            disabled=False
        )

        self.watermark_opacity = widgets.IntSlider(
            value=128,
            min=0,
            max=255,
            step=1,
            description='水印透明度:',
            disabled=False
        )

        self.watermark_position = widgets.Dropdown(
            options=['左上', '右上', '左下', '右下', '居中'],
            value='右下',
            description='水印位置:',
            disabled=False
        )

        self.speed_slider = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=5.0,
            step=0.1,
            description='播放速度:',
            disabled=False
        )

        self.crop_left = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            step=1,
            description='左裁剪(%):',
            disabled=False
        )

        self.crop_right = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            step=1,
            description='右裁剪(%):',
            disabled=False
        )

        self.crop_top = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            step=1,
            description='上裁剪(%):',
            disabled=False
        )

        self.crop_bottom = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            step=1,
            description='下裁剪(%):',
            disabled=False
        )

        self.rotate_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=360,
            step=90,
            description='旋转角度:',
            disabled=False
        )

        self.brightness_slider = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=3.0,
            step=0.1,
            description='亮度:',
            disabled=False
        )

        self.contrast_slider = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=3.0,
            step=0.1,
            description='对比度:',
            disabled=False
        )

        self.preview_button = widgets.Button(
            description='预览效果',
            disabled=False,
            button_style='',
            tooltip='预览处理后的GIF'
        )

        self.download_button = widgets.Button(
            description='下载GIF',
            disabled=True,
            button_style='',
            tooltip='下载处理后的GIF'
        )

        self.output = widgets.Output()
        self.processed_gif = None

        # 设置事件处理
        self.upload.observe(self.on_upload_change, names='value')
        self.width_slider.observe(self.on_size_change, names='value')
        self.height_slider.observe(self.on_size_change, names='value')
        self.keep_aspect.observe(self.on_keep_aspect_change, names='value')
        self.preview_button.on_click(self.on_preview_click)
        self.download_button.on_click(self.on_download_click)

    def on_upload_change(self, change):
        if len(self.upload.value) > 0:
            self.download_button.disabled = False
            self.process_gif()

    def on_size_change(self, change):
        if self.keep_aspect.value and len(self.upload.value) > 0:
            if change['name'] == 'value':
                if change['owner'] == self.width_slider:
                    original_gif = Image.open(io.BytesIO(list(self.upload.value.values())[0]['content']))
                    original_width, original_height = original_gif.size
                    aspect_ratio = original_height / original_width
                    new_height = int(self.width_slider.value * aspect_ratio)
                    self.height_slider.unobserve(self.on_size_change, names='value')
                    self.height_slider.value = new_height
                    self.height_slider.observe(self.on_size_change, names='value')
                elif change['owner'] == self.height_slider:
                    original_gif = Image.open(io.BytesIO(list(self.upload.value.values())[0]['content']))
                    original_width, original_height = original_gif.size
                    aspect_ratio = original_width / original_height
                    new_width = int(self.height_slider.value * aspect_ratio)
                    self.width_slider.unobserve(self.on_size_change, names='value')
                    self.width_slider.value = new_width
                    self.width_slider.observe(self.on_size_change, names='value')

    def on_keep_aspect_change(self, change):
        if change['new'] and len(self.upload.value) > 0:
            self.on_size_change({'name': 'value', 'owner': self.width_slider})

    def on_preview_click(self, button):
        if len(self.upload.value) > 0:
            self.process_gif()

    def on_download_click(self, button):
        if self.processed_gif is not None:
            from google.colab import files
            with io.BytesIO() as output_buffer:
                self.processed_gif.save(output_buffer, format='GIF', save_all=True)
                files.download(output_buffer.getvalue())

    def process_gif(self):
        with self.output:
            clear_output(wait=True)
            if len(self.upload.value) == 0:
                print("请先上传GIF文件")
                return

            # 获取上传的GIF
            uploaded_file = list(self.upload.value.values())[0]
            gif_bytes = uploaded_file['content']

            # 打开GIF
            original_gif = Image.open(io.BytesIO(gif_bytes))

            # 获取GIF的帧
            frames = []
            for frame in ImageSequence.Iterator(original_gif):
                frames.append(frame.copy())

            # 处理每一帧
            processed_frames = []
            for frame in frames:
                # 裁剪
                width, height = frame.size
                left = int(width * self.crop_left.value / 100)
                right = width - int(width * self.crop_right.value / 100)
                top = int(height * self.crop_top.value / 100)
                bottom = height - int(height * self.crop_bottom.value / 100)

                if left < right and top < bottom:
                    frame = frame.crop((left, top, right, bottom))

                # 调整大小
                frame = frame.resize((self.width_slider.value, self.height_slider.value))

                # 旋转
                if self.rotate_slider.value != 0:
                    frame = frame.rotate(self.rotate_slider.value, expand=True)

                # 调整亮度和对比度
                if self.brightness_slider.value != 1.0 or self.contrast_slider.value != 1.0:
                    from PIL import ImageEnhance
                    if self.brightness_slider.value != 1.0:
                        enhancer = ImageEnhance.Brightness(frame)
                        frame = enhancer.enhance(self.brightness_slider.value)
                    if self.contrast_slider.value != 1.0:
                        enhancer = ImageEnhance.Contrast(frame)
                        frame = enhancer.enhance(self.contrast_slider.value)

                # 添加水印
                if self.watermark_text.value:
                    draw = ImageDraw.Draw(frame)
                    try:
                        font = ImageFont.truetype("arial.ttf", self.watermark_size.value)
                    except:
                        font = ImageFont.load_default()

                    text = self.watermark_text.value
                    text_width, text_height = draw.textsize(text, font=font)

                    margin = 10
                    if self.watermark_position.value == '左上':
                        position = (margin, margin)
                    elif self.watermark_position.value == '右上':
                        position = (frame.width - text_width - margin, margin)
                    elif self.watermark_position.value == '左下':
                        position = (margin, frame.height - text_height - margin)
                    elif self.watermark_position.value == '右下':
                        position = (frame.width - text_width - margin, frame.height - text_height - margin)
                    else:  # 居中
                        position = ((frame.width - text_width) // 2, (frame.height - text_height) // 2)

                    # 创建一个带有透明度的水印
                    watermark = Image.new('RGBA', frame.size, (0, 0, 0, 0))
                    watermark_draw = ImageDraw.Draw(watermark)
                    watermark_draw.text(position, text, font=font, fill=(255, 255, 255, self.watermark_opacity.value))

                    # 合并水印和原始图像
                    frame = Image.alpha_composite(frame.convert('RGBA'), watermark).convert('RGB')

                processed_frames.append(frame)

            # 保存处理后的GIF
            with io.BytesIO() as output_buffer:
                # 计算帧持续时间，调整播放速度
                duration = original_gif.info.get('duration', 100)
                duration = int(duration / self.speed_slider.value)

                processed_frames[0].save(
                    output_buffer,
                    format='GIF',
                    save_all=True,
                    append_images=processed_frames[1:],
                    duration=duration,
                    loop=0
                )

                self.processed_gif = Image.open(output_buffer)
                display(self.processed_gif)

    def display(self):
        # 创建布局
        size_controls = widgets.HBox([self.width_slider, self.height_slider, self.keep_aspect])
        crop_controls = widgets.HBox([self.crop_left, self.crop_right])
        crop_controls2 = widgets.HBox([self.crop_top, self.crop_bottom])
        watermark_controls = widgets.HBox([
            self.watermark_text,
            self.watermark_size,
            self.watermark_opacity,
            self.watermark_position
        ])
        adjustment_controls = widgets.HBox([
            self.speed_slider,
            self.rotate_slider,
            self.brightness_slider,
            self.contrast_slider
        ])
        button_controls = widgets.HBox([self.preview_button, self.download_button])

        # 显示所有控件
        display(widgets.VBox([
            self.upload,
            widgets.Label('调整大小:'),
            size_controls,
            widgets.Label('裁剪:'),
            crop_controls,
            crop_controls2,
            widgets.Label('水印设置:'),
            watermark_controls,
            widgets.Label('其他调整:'),
            adjustment_controls,
            button_controls,
            self.output
        ]))


# 修复 ImageSequence 导入问题
try:
    from PIL import ImageSequence
except ImportError:
    class ImageSequence:
        @staticmethod
        def Iterator(image):
            try:
                while True:
                    yield image.copy()
                    image.seek(image.tell() + 1)
            except EOFError:
                pass

# 创建并显示处理器
processor = GIFProcessor()
processor.display()