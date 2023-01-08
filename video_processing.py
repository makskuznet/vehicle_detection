import os
import subprocess
import cv2


def download_fragment(url):
    if os.path.exists('traffik.mp4'):
        os.remove('traffik.mp4')

    process_get_m3u8 = subprocess.run(f'youtube-dl -g "{url}"', stdout=subprocess.PIPE)
    print('> Загрузка видео началась')
    # stdout и stderr отвечают за то, куда выводится текст с консоли, timeout в секундах, можно сделать время начала и конца загрузки (-ss,-to)
    download_process = subprocess.run(
        f'ffmpeg -i "{process_get_m3u8.stdout.decode("utf-8")}" -t 00:2:00.00 -c copy traffik.mp4',
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True, timeout=1800)
    if download_process.returncode == 0:
        print('> Файл загружен')


def split_video_by_frames(step=1):
    """функция постановки детекторов, возвращает коориднаты каждого из них"""
    vidcap = cv2.VideoCapture('traffik.mp4')
    FPS = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / FPS

    print('fps = ' + str(FPS))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    minutes = int(duration / 60)
    seconds = duration % 60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    # *step = int(input('Введите шаг чтения кадров = '))
    # *radius = int(input('Введите радиус ячейки = '))
    # step = 1
    radius = 17
    success, image = vidcap.read()
    count = 0
    if success:
        cv2.imwrite('frames/frame%d.png' % count, image)
        count += 1
    while success:
        success, image = vidcap.read()
        if (count % step == 0) and (success):
            cv2.imwrite('frames/frame%d.png' % count, image)  # save frame as JPEG file
        count += 1
    print('> Разбивка по кадрам завершена')
    vidcap.release()  # освобождаем память
    return FPS
    # старая версия преобразования кадров в серый
    '''rgb_images = os.listdir('frames')  # получаем список всех rgb изображений в папке
    for rgb_img in rgb_images:  # в цикле сохраняем все цветные изображения чёрно-белыми
        img = Image.open('frames/%s' % rgb_img).convert('LA')
        img.save('frames/%s' % rgb_img.replace('_', ''))
        os.remove('frames/%s' % rgb_img)  # удаляем поочерёдно rgb изображения
    print('> Все кадры преобразованы в Grayscale')'''


def save_preview(detectors_coords: list, radius=20):
    """Сохраняем первый фрейм с расставленными детекторами"""

    frame0 = cv2.imread('frames/frame0.png', cv2.IMREAD_GRAYSCALE)  # первый фрейм для отметки детекторов
    for detector_num in range(len(detectors_coords[0])):
        x = detectors_coords[0][detector_num]
        y = detectors_coords[1][detector_num]
        frame0 = cv2.rectangle(frame0, (x - radius, y - radius), (x + radius, y + radius), (255, 0, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame0, 'det' + str(detector_num) + ':' + str(x) + ',' +
                    str(y), (x + radius + 2, y), font,
                    0.5, (255, 255, 255), 1)
    cv2.imwrite('frame_with_detectors.png', frame0)
