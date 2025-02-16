import kagglehub

path = kagglehub.dataset_download(
    "hunhnguynphc/spanish-to-english-dataset"
)
print(path) #手动把文件移到当前目录