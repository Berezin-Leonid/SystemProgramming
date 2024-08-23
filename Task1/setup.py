from setuptools import setup, find_packages

setup(
    name='ECG_RESNET',  # Имя вашего пакета
    version='0.1.0',  # Версия вашего пакета
    description='Detecting AFIB with convolution neural network',  # Краткое описание пакета
    long_description=open('README.md').read(),  # Длинное описание пакета (обычно из файла README.md)
    long_description_content_type='text/markdown',  # Тип содержимого длинного описания
    author='Berezin Leonid',  # Ваше имя
    author_email='berezinlion@gmail.com',  # Ваш email
    url='https://github.com',  # URL вашего проекта (например, на GitHub)
    packages=find_packages(),  # Найти все пакеты в проекте
    install_requires=[
        'numpy',  # Зависимости вашего пакета
        'torch',
        'scikit-learn'
        'pandas'
        'wfdb',
        'ast',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Минимальная версия Python
)
