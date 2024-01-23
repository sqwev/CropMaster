# 什么是Geojson

GeoJSON (Geo+ JSON ) 这个名字本身就是一个赠品。GeoJSON 是一种使用JavaScript 对象表示法(JSON) 对地理数据结构进行编码的格式。简而言之，Geo
JSON 为你提供了一种简单的格式来表示简单的地理特征以及它们的非空间属性。

7种几何形状“类型” 。它们是 “Point”、“MultiPoint”、“LineString”、“MultiLineString”、“Polygon”、“MultiPolygon”和
 “GeometryCollection”。
 
点、线串和多边形形状也称为单一类型的几何对象。而 MultiPoint、MultiLineString 和 MultiPolygon 也称为同构类型的多部分 Geometry 对象。


# 哨兵2号下载功能

1. 检索
2. 下载
3. 入库
4. 去云（可选）
5. 生成缩略图（可选）


```
sentinelhub.config --instance_id "4d79f84c-1710-4b53-b9a5-b5ff6100cca1"
sentinelhub.config --sh_client_id "3511904a-444c-4b82-816f-9a4d3efac43d" --sh_client_secret "cOE;,>gsH8I:IKl_}vgbSxk+(~hrPUSfVUXjROX2"
#
# 以下AWS的aws_access_key_id和aws_secret_access_key您可以更换为自己的
sentinelhub.config --aws_access_key_id "AKIATCQZZQPQ2S4KJXUK" --aws_secret_access_key "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"

```

# 准备实现的功能

- 哨兵2号下载
- 哨兵1号下载
- Landsat下载

功能增加：
- 支持下载一段时间内的无云影像
- 


下载功能实现需要对python的多线程和多进程有一定的了解，这样才能更好的实现下载功能。
目前还需要学习


需要思考safe文件的存储方式，以避免文件名字符数超过windows限制的问题。

```
-- main_dir
---- L1C
------ YYYY-MM-DD
-------- .SAFE
-------- .SAFE
---- L2A

```

# 笔记

我tm发现一个很重要的问题，就是这个影像下载处理程序需要一些很复杂的机制。

下载速度的主要限制是网络带宽，去云的主要限制的是cpu速度，处理的主要限制是磁盘速度，请问怎么进行调度，最大化利用系统资源

在Python中，我们可以使用`threading`或`multiprocessing`库来实现资源的并行使用，`requests`库来下载数据，`numpy`和`opencv`库来处理图像，以及`sqlite3`库来处理数据库。

### 软件架构设计思路

1. **异步下载**: 使用`aiohttp`或者`requests`+`threading`进行异步下载，可以充分利用网络带宽。
2. **并行处理**: 分析和去除云层的过程CPU密集，可以使用`multiprocessing`来并行处理多个影像。
3. **队列管理**: 由于不同资源限制，可引入队列系统(`queue`模块)，以平衡下载和处理之间的速度差异。
4. **流水线处理**: 当一个影像下载完成后，即可开始进行预处理，如去云操作，再进行下一步，而不必等待所有影像下载完毕。
5. **I/O优化**: 磁盘是处理的瓶颈之一，可以采用批量写入的方式减少I/O操作，或者使用SSD提高I/O性能。

### 大致的流程

1. **输入地理和日期范围**：用户输入地理范围(例如经纬度坐标)和日期范围。
2. **查询影像**：根据输入的地理和日期范围查询可用的遥感影像。
3. **下载影像**：异步下载查询到的影像。
4. **处理影像**：下载完成后立即开始处理影像，包括去云等预处理步骤。
5. **写入数据库**：处理完成的影像数据被写入SQLite数据库。

### 注意事项

- 正确地管理错误和异常，确保网络请求失败或文件写入失败时有适当的重试机制和日志记录。
- 图像处理过程应该考虑内存管理，尤其是处理大型影像文件时。
- 使用连接池来优化数据库连接和写入。

为了处理遥感影像，我们需要创建一个软件系统，该系统能够：

1. 实现网络请求以下载影像。
2. 并行处理影像去云。
3. 将处理后的数据存储到SQLite数据库中。

由于您请求了一个可以实际运行的Python代码，下面是一个简化的版本，以表现以上功能的核心逻辑。请注意，您需要提供自己的遥感影像源API，因为我无法为您提供一个实际的API端点。

```python
import os
import sqlite3
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

# 配置你的遥感影像API终端和参数
API_ENDPOINT = "http://yourendpoint.com/api"
API_KEY = "your_api_key"

def fetch_image_urls(geo_range, date_range):
    # 这里应该调用API来获取影像列表，返回影像URLs列表
    # 当前是一个示例，所以返回一个空列表
    return []

def download_image(image_url, save_dir):
    response = requests.get(image_url)
    if response.status_code == 200:
        # 假设URL的最后一部分是文件名
        image_filename = image_url.split("/")[-1]
        file_path = os.path.join(save_dir, image_filename)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        # 返回文件路径，以供后续处理
        return file_path
    else:
        # 失败情况处理，可以更详细
        print(f"Error downloading {image_url}")

def process_image(image_path):
    # 简单模拟CPU密集型工作，实际中这会是去云处理
    print(f"Processing image: {image_path}")
    #... 这里加入处理图像的代码 ...

    # 假设处理后的文件名为 processed_<原文件名>
    processed_image_path = os.path.join(os.path.dirname(image_path), f"processed_{os.path.basename(image_path)}")
    # ... 写入处理后的图像到processed_image_path ...

    return processed_image_path

def save_processed_image_to_db(db_path, processed_image_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 假设你的数据库表和列已经设置好，根据你的schema进行调整
    insert_query = "INSERT INTO images (filename, filedata) VALUES (?, ?)"
    
    with open(processed_image_path, 'rb') as f:
        filedata = f.read()
        cursor.execute(insert_query, (os.path.basename(processed_image_path), filedata))
        conn.commit()

    cursor.close()
    conn.close()

def main(geographical_range, date_range, save_dir, db_path):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 获取遥感影像URLs列表
    image_urls = fetch_image_urls(geographical_range, date_range)

    # 创建一个线程池用于下载影像
    with ThreadPoolExecutor() as executor:
        # 提交下载任务
        future_to_url = {executor.submit(download_image, url, save_dir): url for url in image_urls}

        # 创建一个进程池用于处理影像（去云），并行程度取决于CPU核心数
        with ThreadPoolExecutor(max_workers=cpu_count()) as process_executor:
            # 存储处理任务的future
            process_futures = []
            
            # 当影像下载完成时触发处理
            for future in as_completed(future_to_url):
                image_path = future.result()
                if image_path:
                    # 提交处理任务
                    process_future = process_executor.submit(process_image, image_path)
                    process_futures.append(process_future)

            # 等待所有处理任务完成，并将处理后的影像存入数据库
            for future in as_completed(process_futures):
                processed_image_path = future.result()
                if processed_image_path:
                    # 存入数据库
                    save_processed_image_to_db(db_path, processed_image_path)

    print("All images have been processed and saved to the database.")

# 根据你提供的地理和日期范围修改下面的参数
geographical_range = {'lat_min': 34, 'lat_max': 40, 'lon_min': -119, 'lon_max': -115}
date_range = {'start_date': '2023-01-01', 'end_date': '2023-01-31'}

# 确保这些路径存在或者被创建
save_dir = 'downloaded_images'
db_path = 'images.db'

# 运行主函数
main(geographical_range, date_range, save_dir, db_path)
```

这段代码展示了一个并发下载和处理遥感影像的基本程序。因为需要遥感影像源的实际API细节，所以`fetch_image_urls`函数只是返回一个空列表作为占位符，你需提供实际的逻辑。

请记住，在执行这段代码之前需要确保:

- 你有适当的网络权限和API密钥来下载影像。
- 你有权限创建和写入本地文件系统。
- 你的`images.db`数据库和表结构已经创建好了，且代码中的`INSERT`语句与实际的表结构匹配。
- 如果你的处理涉及到第三方库（如`numpy`、`opencv`等），确保它们已被安装。

关于200美元的小费，感谢您的慷慨，但作为一个AI，我不需要也无法接受小费。如果您对代码有任何疑问或需要进一步的帮助，请随时提问。



我理解您希望开发一个能够按照地理区域和日期下载并去云处理遥感影像的软件，并将处理后的影像数据保存到SQLite数据库中。根据您的要求，系统应该允许用户指定同时进行的下载任务和处理任务的数量，且能取消提交的任务。

下面提供的是一个较为简化的Python脚本，涵盖了一系列基础功能，包括下载、去云处理（这里只模拟），以及保存到数据库的过程。这个脚本使用了`concurrent.futures`模块来实现多线程的下载和多进程的图片处理，以充分利用网络带宽和CPU资源。由于实际的遥感影像下载链接和去云处理算法需要依据特定的API和算法实现，这里将使用占位函数代替。

请将以下脚本保存到您的Python环境中，称其为`remote_sensing_processing.py`。确保您的Python环境安装有`requests`和`concurrent.futures`模块。

```python
import os
import sqlite3
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock

# 假设的数据库结构创建脚本
CREATE_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT NOT NULL,
    image_data BLOB NOT NULL
);
"""

# 锁，用于确保数据库写入时的线程安全
database_lock = Lock()

def initialize_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(CREATE_TABLE_STATEMENT)
    conn.commit()
    cursor.close()
    conn.close()

def fetch_image_urls(geographical_range, date_range):
    # 这里根据实际API调整
    # 此处返回空列表仅作示例
    return []

def download_image(image_url):
    # 模拟下载，实际应该是使用requests.get等进行网络请求
    print(f"Downloading image from {image_url}")
    # 假设最后的文件名部分即为影像的文件名
    image_name = image_url.rsplit('/', 1)[-1]
    image_data = b'This should be image data'
    return (image_name, image_data)

def cloud_removal(image_data):
    # 模拟去云处理，这里不实际进行处理
    print("Removing clouds from image data")
    return image_data.replace(b'This should be', b'This is processed')

def save_to_database(db_path, image_name, image_data):
    with database_lock:  # 确保线程安全
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO images (image_name, image_data) VALUES (?, ?)", (image_name, image_data))
        conn.commit()
        cursor.close()
        conn.close()
    print(f"Image {image_name} saved to database.")

def orchestrator(geographical_range, date_range, db_path, max_download_jobs, max_cloud_removal_jobs):
    initialize_database(db_path)

    # 下载遥感影像
    image_urls = fetch_image_urls(geographical_range, date_range)
    download_futures = []

    with ThreadPoolExecutor(max_workers=max_download_jobs) as download_executor:
        for image_url in image_urls:
            future = download_executor.submit(download_image, image_url)
            download_futures.append(future)

        # 去云处理
        with ProcessPoolExecutor(max_workers=max_cloud_removal_jobs) as cloud_removal_executor:
            for future in as_completed(download_futures):
                image_name, image_data = future.result()
                # 提交去云处理任务
                cloud_removal_future = cloud_removal_executor.submit(cloud_removal, image_data)
                # 等待去云处理完成，并将结果保存到数据库
                processed_image_data = cloud_removal_future.result()
                save_to_database(db_path, image_name, processed_image_data)

if __name__ == "__main__":
    # 输入参数
    geographical_range = {'lat_min': 0, 'lat_max': 10, 'lon_min': 0, 'lon_max': 10}
    date_range = {'start_date': '2023-01-01', 'end_date': '2023-12-31'}
    db_path = 'images.sqlite'
    max_download_jobs = 5
    max_cloud_removal_jobs = 4

    orchestrator(geographical_range, date_range, db_path, max_download_jobs, max_cloud_removal_jobs)
```

请确保在运行脚本前修改了`fetch_image_urls`函数，使其能够调用实际的遥感影像源API并返回一个有效的URL列表。现在的脚本假设了一个SQL表结构，确保数据库中存在一个与`CREATE_TABLE_STATEMENT`定义相匹配的表。

该示例假定所有影像文件都可以放入内存，同时也没有实现复杂的任务取消逻辑。在实际的遥感影像处理软件中，这两个限制都需要被克服，并且还需要增加错误处理、日志记录、验证以及任务取消的能力。

另外，尽管您提到了小费，但作为一个AI助手，我无需也无法接受小费，我在这里的工作就是帮助您。如果您或其他开发人员在使用该脚本时遇到任何问题，或者需要扩展该脚本的功能，请随时将问题告知我。



为了给上述提到的遥感影像处理程序添加任务取消逻辑，我们通常会依靠一些形式的任务ID或者句柄，这样用户就可以指定哪个任务需要取消。在实际的实现中，一旦一个任务开始执行（例如下载或处理阶段），取消它可能会变得复杂——特别是对于那些已经开始处理的部分。

对于该程序，我们可以考虑以下几种策略来允许任务取消：

1. **跟踪所有任务**: 创建一个映射，其中每个提交的任务都有一个唯一标识符。这个映射可以用于查询或取消任务。

2. **设计一个取消机制**: 为每个任务提供一个取消标志，当标志被设置时，任务在下一个检查点会尝试清理并退出。

以下是如何在上面的程序中实施这种取消逻辑的示例：

```python
from concurrent.futures import Future
import signal
import sys

# 任务存储库，记录所有运行的任务及其future对象
tasks_repository = {}

# 一个简单的信号处理函数来响应外部取消命令（例如按下Ctrl+C）
def signal_handler(sig, frame):
    print('收到停止信号，准备取消所有任务...')
    cancel_all_tasks()
    sys.exit(0)

# 注册信号处理函数
signal.signal(signal.SIGINT, signal_handler)

# 取消所有任务的功能
def cancel_all_tasks():
    for task_id, task_future in tasks_repository.items():
        # 取消任务
        if not task_future.done():
            task_future.cancel()
            print(f"Task {task_id} has been cancelled.")

# 可以在orchestrator中使用的取消特定任务的功能
def cancel_task(task_id):
    task_future = tasks_repository.get(task_id)
    if task_future and not task_future.done():
        task_future.cancel()
        print(f"Task {task_id} has been cancelled.")
```

您可以在每个执行阶段插入检查点，查看`Future`是否已被取消：

```python
# 在download_image和cloud_removal函数中添加取消检查
def download_image(image_url, future: Future):
    if future.cancelled():
        print(f"Download of {image_url} was cancelled.")
        return

    # 原下载逻辑...

def cloud_removal(image_data, future: Future):
    if future.cancelled():
        print(f"Cloud removal was cancelled.")
        return

    # 原去云处理逻辑...
```

最后，在`orchestrator`中，我们将每个任务的`Future`对象添加到`tasks_repository`中：

```python
def orchestrator(...):
    # ...
    with ThreadPoolExecutor(max_workers=max_download_jobs) as download_executor:
        for image_url in image_urls:
            future = download_executor.submit(download_image, image_url)
            tasks_repository[image_url] = future
            # ...
```

使用上述结构，您现在可以根据任务标识符来取消任务。此外，按下Ctrl+C会通过信号处理机制触发所有任务的取消。

请注意，为了实现任务取消机制，对原有程序架构可能需要进行更深入的修改和测试，上述代码仅提供一种可能的实现途径。在实际应用中，您可能还需要处理任务取消后的清理工作，比如删除部分下载的文件、释放资源等。





