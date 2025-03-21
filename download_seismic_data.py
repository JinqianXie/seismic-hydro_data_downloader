import os
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, List, Dict
from obspy import UTCDateTime, read, read_inventory, Trace, Stream
from obspy.clients.fdsn import Client
from obspy.signal.filter import bandpass
from obspy.core import AttribDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import glob
import numpy as np


@dataclass
class ProcessingConfig:
    """
    数据处理配置类

    属性:
        freqmin (float): 滤波最小频率 (Hz)
        freqmax (float): 滤波最大频率 (Hz)
        max_spike_width (int): 尖灭处理的最大宽度(采样点数)
        spike_threshold (float): 毛刺检测阈值(标准差的倍数)
    """
    freqmin: float = 2.0
    freqmax: float = 8.0
    max_spike_width: int = 10
    spike_threshold: float = 3.0


class SeismicDataProcessor:
    """地震数据处理类"""

    @staticmethod
    def remove_mean(tr: Trace) -> Trace:
        """去均值"""
        tr.detrend('demean')
        return tr

    @staticmethod
    def remove_trend(tr: Trace) -> Trace:
        """去趋势"""
        tr.detrend('linear')
        return tr

    @staticmethod
    def despike(tr: Trace, max_width: int, threshold: float) -> Trace:
        """
        去毛刺

        参数:
            tr: 地震数据
            max_width: 毛刺最大宽度
            threshold: 检测阈值(标准差的倍数)
        """
        data = tr.data
        std = np.std(data)
        spikes = np.where(np.abs(data) > threshold * std)[0]

        for spike in spikes:
            left = max(0, spike - max_width)
            right = min(len(data), spike + max_width + 1)
            data[spike] = np.median(data[left:right])

        return tr

    @staticmethod
    def taper(tr: Trace, max_percentage: float = 0.05) -> Trace:
        """
        信号尖灭处理

        参数:
            tr: 地震数据
            max_percentage: 信号两端处理的最大百分比
        """
        tr.taper(max_percentage)
        return tr

    @staticmethod
    def bandpass_filter(tr: Trace, freqmin: float, freqmax: float) -> Trace:
        """
        带通滤波

        参数:
            tr: 地震数据
            freqmin: 最小频率 (Hz)
            freqmax: 最大频率 (Hz)
        """
        tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4,
                  zerophase=True)
        return tr


@dataclass
class DownloadConfig:
    """
    地震数据下载配置类

    属性:
        network (str): 地震台网代码，例如 "IM"
        station (str): 台站代码，支持通配符，默认为 "H11??"
        location (str): 位置代码，默认为 "*"
        channel (str): 通道代码，默认为 "EDH"
        data_type (str): 数据类型，可选 'raw'（原始数据）或 'processed'（去除仪器响应），默认为 'raw'
        max_retries (int): 下载失败时的最大重试次数，默认为 3
        max_workers (int): 并行下载的最大线程数，默认为 5
        retry_delay_base (int): 重试延迟的基数（实际延迟为 base^retry_count），默认为 2
        single_thread (bool): 是否使用单线程模式，默认为 False
        need_station_coords (bool): 是否需要处理台站位置信息，默认为 False
        merge_after_download (bool): 是否在下载后立即合并数据段，默认为 False
        merge_method (str): 合并方法，可选 "fill_value"（填充值）或 "interpolate"（插值），默认为 "fill_value"
        merge_fill_value (float): 合并时使用的填充值，默认为 0
        response_output (str): 去除仪器响应后输出的类型，可选 'VEL'(速度)、'DISP'(位移)或 'ACC'(加速度)，默认为 'VEL'
    """
    network: str
    station: Union[str, List[str]] = "H11??"  # 可以是单个字符串或字符串列表
    location: str = "*"
    channel: str = "EDH"
    data_type: str = 'raw'
    max_retries: int = 3
    max_workers: int = 5
    retry_delay_base: int = 2
    single_thread: bool = False
    need_station_coords: bool = False  # 添加标志，控制是否需要处理台站位置信息
    merge_after_download: bool = False  # 下载后立即合并数据段
    merge_method: str = "fill_value"  # 新增参数：合并方法
    merge_fill_value: float = 0       # 新增参数：填充值
    # 去除仪器响应后输出的类型: 'VEL'(速度), 'DISP'(位移), 'ACC'(加速度)
    response_output: str = 'VEL'


@dataclass
class SeismicDataDownloader:
    """
    地震数据下载器主类

    负责处理地震数据的下载、保存和日志记录。支持多线程下载、错误重试、
    仪器响应去除等功能。

    属性:
        config (DownloadConfig): 下载配置对象
        save_path (Path): 数据保存路径
        client (Client): FDSN客户端实例
        logger (Logger): 日志记录器实例
    """

    def __init__(self, config: DownloadConfig, save_path: Union[str, Path],
                 log_file: str = "data_download.log"):
        """
        初始化下载器

        参数:
            config (DownloadConfig): 下载配置对象，包含网络、台站等信息
            save_path (Path): 数据保存的根目录
            log_file (Logger): 日志文件路径，默认为 "data_download.log"
        """
        self.config = config
        self.save_path = Path(save_path)
        self.client = Client("IRIS")  # 初始化FDSN客户端，使用IRIS服务

        # 将日志文件放在 save_path 下
        log_path = self.save_path / log_file

        self._setup_logging(str(log_path))
        self._setup_directories()

        self.processor = SeismicDataProcessor()
        self.process_config = ProcessingConfig()  # 使用默认配置

    def _setup_logging(self, log_file: str) -> None:
        """
        配置日志系统

        设置日志格式、级别和输出文件。记录包括时间戳、日志级别和具体消息。

        参数:
            log_file: 日志文件的保存路径
        """
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_directories(self) -> None:
        """
        创建必要的目录结构

        确保数据保存的根目录存在，如果不存在则创建
        """
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _get_day_folders(self, day: datetime.datetime) -> tuple[Path, Path]:
        """
        获取指定日期的数据和响应文件保存目录

        为每一天的数据创建单独的目录结构，包括数据目录和仪器响应文件目录

        参数:
            day: 日期对象

        返回:
            tuple: (数据目录路径, 响应文件目录路径)
        """
        day_str = day.strftime("%Y-%m-%d")
        day_folder = self.save_path / day_str  # 数据主目录
        response_folder = day_folder / "responses"  # 仪器响应文件目录

        # 确保两个目录都存在
        day_folder.mkdir(exist_ok=True)
        response_folder.mkdir(exist_ok=True)

        return day_folder, response_folder

    def _download_instrument_response(self, day: datetime.datetime,
                                      response_folder: Path,
                                      station: str) -> tuple[Optional[str], Optional[dict]]:
        """
        下载仪器响应文件并提取台站位置信息

        获取指定日期的仪器响应信息，并保存为StationXML格式

        修改仪器响应下载函数以接受具体台站

        参数:
            day: 日期对象
            response_folder: 响应文件保存目录

        返回:
        tuple: (仪器响应对象, 台站位置信息字典)
        """
        try:
            # 从FDSN服务获取仪器响应数据
            # 获取台站元数据（包括位置信息）
            inventory = self.client.get_stations(
                network=self.config.network,
                # station=self.config.station,
                station=station,  # 使用具体台站
                location=self.config.location,
                channel=self.config.channel,
                starttime=day,
                endtime=day + datetime.timedelta(days=1),
                level="response"  # 请求完整的仪器响应信息
            )

            # 构造响应文件名并保存
            inventory_file = response_folder / \
                f"{self.config.network}_{day.strftime('%Y-%m-%d')}_response.xml"
            inventory.write(str(inventory_file), format="STATIONXML")
            self.logger.info(f"仪器响应文件已保存到 {inventory_file}")

            # 提取台站位置信息
            station_coords = {}
            for network in inventory:
                for station in network:
                    station_id = station.code
                    station_coords[station_id] = {
                        'latitude': station.latitude,
                        'longitude': station.longitude,
                        'elevation': station.elevation
                    }

            return inventory, station_coords

        except Exception as e:
            self.logger.error(f"下载仪器响应文件失败: {e}")
            return None, None

    def _process_and_save_traces(self, st, inventory, station_coords: dict,
                                 day_folder: Path, day_str: str) -> None:
        """
        处理和保存波形数据，并添加台站位置信息

        根据配置处理波形数据（可选去除仪器响应），并保存为SAC格式

        参数:
            st: 波形数据流对象
            inventory: 仪器响应信息
            day_folder: 数据保存目录
            day_str: 日期字符串 (YYYY-MM-DD)
        """
        # 如果需要去除仪器响应且有响应数据
        if self.config.data_type != 'raw' and inventory:
            # 根据配置选择输出类型：速度、位移或加速度
            if self.config.response_output == 'DISP':
                output = "DISP"  # 位移
            elif self.config.response_output == 'VEL':
                output = "VEL"   # 速度
            elif self.config.response_output == 'ACC':
                output = "ACC"   # 加速度
            else:
                output = "VEL"   # 默认使用速度

            try:
                self.logger.info(f"去除仪器响应，输出类型: {output}")
                st.remove_response(inventory=inventory, output=output)
            except Exception as e:
                self.logger.error(f"去除仪器响应失败: {e}")

        # 对每个台站创建一个计数器字典，用于跟踪片段
        segment_counters = {}

        # 保存每个通道的数据
        for tr in st:
            station_id = tr.stats.station
            channel_id = tr.stats.channel

            # 初始化台站-通道组合的计数器
            station_channel_key = f"{station_id}_{channel_id}"
            if station_channel_key not in segment_counters:
                segment_counters[station_channel_key] = 0

            # 获取并增加段计数
            segment_num = segment_counters[station_channel_key]
            segment_counters[station_channel_key] += 1

            # 只在需要时处理台站位置信息
            if self.config.need_station_coords:
                if station_id in station_coords:
                    coords = station_coords[station_id]

                    # 正确创建 SAC 头段
                    if not hasattr(tr.stats, 'sac'):
                        tr.stats.sac = AttribDict()

                    # 将位置信息添加到SAC头段
                    tr.stats.sac.stla = coords['latitude']
                    tr.stats.sac.stlo = coords['longitude']
                    tr.stats.sac.stel = coords['elevation']

                    self.logger.info(f"已添加台站 {station_id} 的位置信息: "
                                     f"lat={coords['latitude']:.4f}, "
                                     f"lon={coords['longitude']:.4f}, "
                                     f"ele={coords['elevation']:.1f}")

            # 为每个片段创建单独的文件名，包含片段编号
            filename = day_folder / \
                f"{self.config.network}.{tr.stats.station}.{tr.stats.channel}.{day_str}.seg{segment_num}.sac"

            # 记录起始和结束时间（可选，帮助识别片段）
            start_time = tr.stats.starttime.strftime("%H%M%S")
            end_time = tr.stats.endtime.strftime("%H%M%S")
            self.logger.info(
                f"保存片段 {segment_num} (时间: {start_time}-{end_time}): {filename}")

            tr.write(str(filename), format="SAC")

    def _fetch_day_data(self, day: datetime.datetime) -> None:
        """
        获取单天的数据

        下载、处理并保存指定日期的地震数据，包含重试机制

        参数:
            day: 日期对象
        """
        day_str = day.strftime("%Y-%m-%d")
        day_folder, response_folder = self._get_day_folders(day)

        # 转换station为列表
        stations = self.config.station if isinstance(
            self.config.station, list) else [self.config.station]

        for station in stations:
            # 重试循环
            for retry in range(self.config.max_retries):
                try:
                    # 获取波形数据
                    st = self.client.get_waveforms(
                        network=self.config.network,
                        station=station,
                        location=self.config.location,
                        channel=self.config.channel,
                        starttime=day,
                        endtime=day + datetime.timedelta(days=1)
                    )

                    # 下载仪器响应并获取台站位置信息，处理数据
                    inventory, station_coords = self._download_instrument_response(
                        day, response_folder, station)  # 传入具体台站
                    self._process_and_save_traces(
                        st, inventory, station_coords, day_folder, day_str)

                    self.logger.info(f"成功处理台站 {station} 在 {day_str} 的数据")
                    break  # 成功处理后跳出重试循环

                except Exception as e:
                    self.logger.error(
                        f"台站 {self.config.network}.{station} 在 {day_str} 的数据处理失败: {e}, "
                        f"重试 ({retry + 1}/{self.config.max_retries})"
                    )
                    if retry < self.config.max_retries - 1:
                        # 使用指数退避策略进行重试
                        delay = self.config.retry_delay_base ** retry
                        time.sleep(delay)
                    else:
                        # 只有在所有重试都失败后才记录
                        self.logger.error(
                            f"达到最大重试次数，放弃台站 {station} 在 {day_str} 的数据")

        # self.logger.error(f"达到最大重试次数，放弃 {day_str} 的数据")
        self.logger.error(f"达到最大重试次数，放弃台站 {station} 在 {day_str} 的数据")

    def download_data(self, starttime: Union[str, UTCDateTime],
                      endtime: Union[str, UTCDateTime]) -> None:
        """
        下载指定时间范围的数据

        主要下载函数，支持多线程并行下载多天数据

        参数:
            starttime: 起始时间，可以是字符串或UTCDateTime对象
            endtime: 结束时间，可以是字符串或UTCDateTime对象
        """
        # 转换时间格式
        if isinstance(starttime, str):
            starttime = UTCDateTime(starttime)
        if isinstance(endtime, str):
            endtime = UTCDateTime(endtime)

        # 生成需要下载的日期列表
        days = [
            starttime + datetime.timedelta(days=i)
            for i in range(int((endtime - starttime) // 86400 + 1))
        ]

        if self.config.single_thread:
            # 单线程模式：顺序处理每一天
            for day in days:
                self._fetch_day_data(day)

                # 如果需要在下载后立即合并数据段
                if self.config.merge_after_download:
                    day_str = day.strftime("%Y-%m-%d")
                    day_folder = self.save_path / day_str
                    self.logger.info(f"下载后立即合并数据: {day_str}")

                    # 先检查文件夹中是否存在文件及其类型
                    seg_files = list(day_folder.glob("*.seg*.sac"))
                    regular_files = list(day_folder.glob("*.sac"))

                    # 优先使用带seg标记的文件，如果没有就使用普通sac文件
                    pattern = "*.seg*.sac" if seg_files else "*.sac"

                    self.merge_daily_segments(
                        day_folder=day_folder,
                        pattern="*.seg*.sac",
                        merge_folder="merged_raw",
                        fill_value=self.config.merge_fill_value,
                        merge_method=self.config.merge_method
                    )
        else:
            # 多线程模式：并行处理多天数据
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_day = {
                    executor.submit(self._fetch_day_data, day): day
                    for day in days
                }

                # 处理完成的任务
                for future in as_completed(future_to_day):
                    day = future_to_day[future]
                    try:
                        future.result()

                        # 如果需要在下载后立即合并数据段
                        if self.config.merge_after_download:
                            day_str = day.strftime("%Y-%m-%d")
                            day_folder = self.save_path / day_str
                            self.logger.info(f"下载后立即合并数据: {day_str}")

                            # 先检查文件夹中是否存在文件及其类型
                            seg_files = list(day_folder.glob("*.seg*.sac"))
                            regular_files = list(day_folder.glob("*.sac"))

                            # 优先使用带seg标记的文件，如果没有就使用普通sac文件
                            pattern = "*.seg*.sac" if seg_files else "*.sac"

                            self.merge_daily_segments(
                                day_folder=day_folder,
                                pattern=pattern,
                                merge_folder="merged_raw",
                                fill_value=self.config.merge_fill_value,
                                merge_method=self.config.merge_method
                            )
                    except Exception as e:
                        self.logger.error(
                            f"处理 {day.strftime('%Y-%m-%d')} 时发生错误: {e}")

    def set_processing_config(self, config: ProcessingConfig) -> None:
        """设置数据处理配置"""
        self.process_config = config

    def _process_single_trace(self, tr: Trace,
                              operations: Dict[str, bool]) -> Optional[Trace]:
        """
        处理单条数据

        参数:
            tr: 地震数据
            operations: 处理操作配置字典
        """
        try:
            # 创建数据副本
            tr_processed = tr.copy()

            # 根据配置执行相应的处理
            if operations.get('remove_mean', True):
                tr_processed = self.processor.remove_mean(tr_processed)

            if operations.get('remove_trend', True):
                tr_processed = self.processor.remove_trend(tr_processed)

            if operations.get('taper', True):
                tr_processed = self.processor.taper(tr_processed)

            if operations.get('despike', True):
                tr_processed = self.processor.despike(
                    tr_processed,
                    self.process_config.max_spike_width,
                    self.process_config.spike_threshold
                )

            if operations.get('bandpass', True):
                tr_processed = self.processor.bandpass_filter(
                    tr_processed,
                    self.process_config.freqmin,
                    self.process_config.freqmax
                )

            return tr_processed

        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            return None

    def remove_response_for_data(self, starttime: Union[str, UTCDateTime],
                                 endtime: Union[str, UTCDateTime],
                                 processed_suffix: str = "_rmresp",
                                 output: str = None,
                                 use_merged: bool = False) -> None:
        """
        为指定时间范围内的数据去除仪器响应
        修改后可处理同一天内的多个数据段

        参数:
            starttime: 起始时间
            endtime: 结束时间
            processed_suffix: 处理后文件的后缀
            output: 输出类型，可选 'VEL'(速度)、'DISP'(位移)或 'ACC'(加速度)，默认使用配置中的值
            use_merged: 是否优先使用合并后的数据
        """
        if isinstance(starttime, str):
            starttime = UTCDateTime(starttime)
        if isinstance(endtime, str):
            endtime = UTCDateTime(endtime)

        # 如果未指定输出类型，使用配置中的值
        if output is None:
            output = self.config.response_output

        days = [starttime + datetime.timedelta(days=i)
                for i in range(int((endtime - starttime) // 86400 + 1))]

        def process_single_day(day: datetime.datetime) -> None:
            day_str = day.strftime("%Y-%m-%d")
            day_folder = self.save_path / day_str
            response_folder = day_folder / "responses"
            processed_folder = day_folder / "processed"
            processed_folder.mkdir(exist_ok=True)

            # 根据use_merged参数决定是否优先查找合并后的数据
            raw_files = []
            if use_merged:
                # 首先尝试从merged_raw文件夹中获取合并后的数据
                merged_folder = day_folder / "merged_raw"
                if merged_folder.exists():
                    raw_files = list(merged_folder.glob(
                        f"{self.config.network}.*.merged.sac"))
                    if raw_files:
                        self.logger.info(
                            f"从合并文件夹 {merged_folder} 中找到 {len(raw_files)} 个文件")
                    else:
                        self.logger.warning(
                            f"在合并文件夹 {merged_folder} 中未找到匹配的文件，将尝试使用原始文件")

            # 如果没有找到合并文件或不使用合并文件，则查找原始文件
            if not raw_files:
                # 尝试不同的文件匹配模式以找到原始数据
                seg_files = list(day_folder.glob(
                    f"{self.config.network}.*.seg*.sac"))
                regular_files = list(day_folder.glob(
                    f"{self.config.network}.*.sac"))

                if seg_files:
                    raw_files = seg_files
                    self.logger.info(f"找到 {len(raw_files)} 个带有段标记的文件")
                else:
                    raw_files = regular_files
                    self.logger.info(f"找到 {len(raw_files)} 个标准SAC文件")

            response_file = list(response_folder.glob(
                f"{self.config.network}_{day_str}_response.xml"))

            if not raw_files:
                self.logger.warning(f"未找到 {day_str} 的原始数据文件")
                return

            if not response_file:
                self.logger.warning(f"未找到 {day_str} 的仪器响应文件")
                return

            response_file = response_file[0]
            inventory = read_inventory(str(response_file))

            for raw_file in raw_files:
                # 保留原始文件名中的段号信息
                # 从文件名中提取基本部分和段号部分
                file_parts = raw_file.stem.split('.')
                station_part = '.'.join(
                    file_parts[:-1]) if len(file_parts) > 1 else file_parts[0]

                # 检查是否有段号或合并标记
                segment_part = ""
                if "seg" in raw_file.stem:
                    # 提取段号部分
                    for part in raw_file.stem.split('.'):
                        if part.startswith("seg"):
                            segment_part = f".{part}"
                            break
                elif "merged" in raw_file.stem:
                    segment_part = ".merged"

                # 构建新的文件名，包含输出类型信息
                processed_file = processed_folder / \
                    f"{station_part}{segment_part}{processed_suffix}_{output.lower()}.sac"

                for retry in range(self.config.max_retries):
                    try:
                        st = read(str(raw_file))
                        # 使用指定的输出类型去除仪器响应
                        self.logger.info(f"去除仪器响应，输出类型: {output}")
                        st.remove_response(inventory=inventory, output=output)
                        st.write(str(processed_file), format="SAC")
                        self.logger.info(
                            f"成功处理文件: {raw_file.name} -> {processed_file.name} (输出类型: {output})")
                        break
                    except Exception as e:
                        self.logger.error(
                            f"处理文件 {raw_file.name} 失败: {e}, 重试 ({retry + 1}/{self.config.max_retries})"
                        )
                        if retry < self.config.max_retries - 1:
                            time.sleep(self.config.retry_delay_base ** retry)
                            continue

                        self.logger.error(f"达到最大重试次数，放弃处理文件 {raw_file.name}")

        # 使用单线程或多线程执行
        if self.config.single_thread:
            for day in days:
                process_single_day(day)
        else:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_day = {executor.submit(
                    process_single_day, day): day for day in days}

                for future in as_completed(future_to_day):
                    day = future_to_day[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(
                            f"处理 {day.strftime('%Y-%m-%d')} 时发生错误: {e}")

    def merge_daily_segments(self, day_folder: Path, pattern: str = "*.sac",
                             merge_folder: str = "merged", fill_value=0,
                             merge_method: str = "fill_value") -> None:
        """
        合并同一天内的不同数据段为连续的数据流

        参数:
            day_folder: 包含数据段的文件夹路径
            pattern: 文件匹配模式，默认为 "*.sac"
            merge_folder: 合并后数据的保存文件夹名称
            fill_value: 填充缺失值使用的值，默认为0
            merge_method: 合并方法，可选 "fill_value"（填充值） 或 "interpolate"（插值）
        """
        merged_folder = day_folder / merge_folder
        merged_folder.mkdir(exist_ok=True)

        # 获取所有匹配的文件 - 打印更多的调试信息
        self.logger.info(f"查找匹配模式: {pattern} 在 {str(day_folder)}")
        all_files = list(day_folder.glob(pattern))

        if not all_files:
            self.logger.warning(f"未找到匹配的文件: {pattern} 在 {str(day_folder)}")

            # 如果是找不到带seg标记的文件，尝试使用不带seg的模式
            if "seg" in pattern:
                alternate_pattern = pattern.replace("seg*.", "")
                self.logger.info(f"尝试使用不带seg的匹配模式: {alternate_pattern}")
                all_files = list(day_folder.glob(alternate_pattern))
                if all_files:
                    self.logger.info(f"使用替代匹配找到 {len(all_files)} 个文件")
                else:
                    # 最后尝试最宽松的匹配
                    self.logger.info("尝试使用最宽松的匹配模式: *.sac")
                    all_files = list(day_folder.glob("*.sac"))
            else:
                # 尝试更宽松的匹配
                self.logger.info("尝试使用更宽松的匹配模式: *.sac")
                all_files = list(day_folder.glob("*.sac"))

            if all_files:
                self.logger.info(f"使用宽松匹配找到 {len(all_files)} 个文件")
            else:
                # 检查目录中是否有任何文件
                all_possible_files = list(day_folder.glob("*"))
                if all_possible_files:
                    self.logger.info(
                        f"目录中有 {len(all_possible_files)} 个文件，但没有匹配的SAC文件。示例: {all_possible_files[:3]}")
                else:
                    self.logger.warning(f"目录 {str(day_folder)} 中没有任何文件")
                return

        self.logger.info(f"找到 {len(all_files)} 个匹配的文件")

        # 按照网络、台站和通道进行分组
        file_groups = {}
        for file_path in all_files:
            # 提取基本标识信息（网络.台站.通道.日期）
            file_name = file_path.name
            parts = file_name.split('.')

            self.logger.debug(f"处理文件: {file_name}, 部分: {parts}")

            # 尝试提取网络、台站、通道信息，更灵活地处理不同的文件命名格式
            network = None
            station = None
            channel = None
            date_part = None

            # 尝试从文件名中提取关键信息
            if len(parts) >= 4:  # 标准命名: network.station.channel.date.其它
                network = parts[0]
                station = parts[1]
                channel = parts[2]
                date_part = parts[3]
            elif len(parts) == 3:  # 简化命名: network.station.channel.其它
                network = parts[0]
                station = parts[1]
                channel = parts[2]
                date_part = "unknown"
            else:
                # 无法识别的格式，使用整个文件名作为组键
                self.logger.warning(f"无法解析文件名: {file_name}, 将整个文件名作为组键")
                group_key = file_name.replace(".sac", "")

                if group_key not in file_groups:
                    file_groups[group_key] = []

                file_groups[group_key].append(file_path)
                continue

            # 用于分组的键
            group_key = f"{network}.{station}.{channel}"
            if date_part != "unknown":
                group_key += f".{date_part}"

            self.logger.debug(f"文件 {file_name} 分配到组: {group_key}")

            if group_key not in file_groups:
                file_groups[group_key] = []

            file_groups[group_key].append(file_path)

        self.logger.info(f"文件分为 {len(file_groups)} 个组")

        # 对每个组进行处理
        for group_key, file_paths in file_groups.items():
            try:
                if len(file_paths) <= 1:
                    self.logger.info(
                        f"组 {group_key} 只有 {len(file_paths)} 个文件，跳过合并")

                    # 如果只有一个文件，可以选择直接复制到合并文件夹
                    if len(file_paths) == 1:
                        output_file = merged_folder / f"{group_key}.merged.sac"
                        original_file = file_paths[0]
                        # 读取并写入，或直接复制
                        try:
                            st = read(str(original_file))
                            st.write(str(output_file), format="SAC")
                            self.logger.info(f"单一文件已复制到 {output_file}")
                        except Exception as e:
                            self.logger.error(f"复制单一文件时出错: {e}")

                    continue

                self.logger.info(f"正在合并组 {group_key} 的 {len(file_paths)} 个数据段")

                # 读取所有段的数据
                stream = Stream()
                for file_path in file_paths:
                    try:
                        segment_stream = read(str(file_path))
                        stream += segment_stream
                        self.logger.debug(f"读取数据段: {file_path.name}")
                    except Exception as e:
                        self.logger.error(f"读取文件 {file_path} 时出错: {e}")

                if len(stream) == 0:
                    self.logger.warning(f"组 {group_key} 没有成功读取任何数据")
                    continue

                # 按通道分组并合并
                merged_stream = Stream()
                channel_ids = set(tr.id for tr in stream)

                self.logger.info(
                    f"组 {group_key} 包含 {len(channel_ids)} 个不同的通道ID: {channel_ids}")

                for channel_id in channel_ids:
                    channel_traces = stream.select(id=channel_id)

                    # 合并前对数据进行排序
                    channel_traces.sort(keys=['starttime'])

                    if len(channel_traces) > 1:
                        self.logger.info(
                            f"通道 {channel_id} 有 {len(channel_traces)} 个片段需要合并")

                        # 输出时间范围信息，用于调试
                        for i, tr in enumerate(channel_traces):
                            self.logger.debug(
                                f"片段 {i}: 开始={tr.stats.starttime}, 结束={tr.stats.endtime}, 持续={tr.stats.endtime - tr.stats.starttime}秒")

                        # 合并设置
                        try:
                            if merge_method == "interpolate":
                                merged_channel = channel_traces.merge(method=1, fill_value=None,
                                                                      interpolation_samples=-1)
                            else:  # 默认使用fill_value
                                merged_channel = channel_traces.merge(
                                    method=1, fill_value=fill_value)

                            merged_stream += merged_channel

                            # 输出合并后的信息
                            for tr in merged_channel:
                                self.logger.info(
                                    f"合并后: 开始={tr.stats.starttime}, 结束={tr.stats.endtime}, 持续={tr.stats.endtime - tr.stats.starttime}秒")

                        except Exception as e:
                            self.logger.error(f"合并通道 {channel_id} 时出错: {e}")
                    else:
                        # 只有一个片段，直接添加
                        self.logger.info(f"通道 {channel_id} 只有一个片段，无需合并")
                        merged_stream += channel_traces

                if len(merged_stream) == 0:
                    self.logger.warning(f"组 {group_key} 没有成功合并任何数据")
                    continue

                # 保存合并后的数据
                output_file = merged_folder / f"{group_key}.merged.sac"

                try:
                    merged_stream.write(str(output_file), format="SAC")
                    self.logger.info(f"成功合并并保存: {output_file}")
                except Exception as e:
                    self.logger.error(f"保存合并数据到 {output_file} 时出错: {e}")

            except Exception as e:
                self.logger.error(f"合并组 {group_key} 时出错: {e}")

    def process_data(self,
                     starttime: Union[str, UTCDateTime],
                     endtime: Union[str, UTCDateTime],
                     operations: Dict[str, bool] = None,
                     input_suffix: str = "",
                     output_suffix: str = "_processed",
                     use_merged: bool = False) -> None:
        """
        处理指定时间范围内的数据
        修改后可处理同一天内的多个数据段

        参数:
            starttime: 起始时间
            endtime: 结束时间
            operations: 处理操作配置字典，包含以下键：
                       - remove_mean: 去均值
                       - remove_trend: 去趋势
                       - taper: 尖灭
                       - despike: 去毛刺
                       - bandpass: 带通滤波
            input_suffix: 输入文件后缀
            output_suffix: 输出文件后缀
            use_merged: 是否优先使用合并后的数据
        """
        # 默认执行所有操作
        if operations is None:
            operations = {
                'remove_mean': True,
                'remove_trend': True,
                'taper': True,
                'despike': True,
                'bandpass': True
            }

        # 转换时间格式
        if isinstance(starttime, str):
            starttime = UTCDateTime(starttime)
        if isinstance(endtime, str):
            endtime = UTCDateTime(endtime)

        days = [
            starttime + datetime.timedelta(days=i)
            for i in range(int((endtime - starttime) // 86400 + 1))
        ]

        def process_single_day(day: datetime.datetime) -> None:
            """处理单天数据"""
            day_str = day.strftime("%Y-%m-%d")
            day_folder = self.save_path / day_str
            processed_folder = day_folder / "processed"
            processed_folder.mkdir(exist_ok=True)

            # 检查是否使用合并后的数据
            if use_merged:
                # 尝试从合并文件夹中获取数据
                merged_folder = day_folder / "merged_raw"
                if merged_folder.exists():
                    if input_suffix:
                        # 带有后缀的合并文件
                        input_pattern = f"*{input_suffix}.merged.sac"
                    else:
                        # 无后缀的合并文件
                        input_pattern = f"{self.config.network}.*.merged.sac"

                    input_files = list(merged_folder.glob(input_pattern))

                    if input_files:
                        self.logger.info(f"从合并文件夹中找到 {len(input_files)} 个文件")
                    else:
                        self.logger.warning(
                            f"在合并文件夹中未找到匹配的文件: {input_pattern}")
                        # 回退到使用未合并的数据
                        use_merged = False
                else:
                    self.logger.warning(f"合并文件夹不存在: {merged_folder}")
                    use_merged = False

            # 如果不使用合并数据或未找到合并数据
            if not use_merged:
                # 查找输入文件，支持带有段号(seg*)的文件
                if input_suffix:
                    # 查找带有特定后缀和段号的文件
                    input_pattern = f"*{input_suffix}.seg*.sac"
                    input_files = list(day_folder.glob(input_pattern))
                    if not input_files:
                        # 尝试在processed文件夹中查找
                        input_files = list(
                            processed_folder.glob(input_pattern))
                    if not input_files:
                        # 兼容旧的文件命名方式
                        input_pattern = f"*{input_suffix}.sac"
                        input_files = list(day_folder.glob(input_pattern))
                        if not input_files:
                            input_files = list(
                                processed_folder.glob(input_pattern))
                else:
                    # 无后缀时的查找模式
                    input_pattern = f"{self.config.network}.*.seg*.sac"
                    input_files = list(day_folder.glob(input_pattern))
                    if not input_files:
                        input_pattern = f"{self.config.network}.*.sac"
                        input_files = list(day_folder.glob(input_pattern))

            if not input_files:
                self.logger.warning(
                    f"未找到符合条件的文件用于处理: {input_pattern} 在 {day_str}")
                return

            for input_file in input_files:
                # 从文件名中提取基本信息并保留段号
                file_stem = input_file.stem
                # if input_suffix and file_stem.endswith(input_suffix):
                #     # 移除输入后缀
                #     file_stem = file_stem[:-len(input_suffix)]

                # 构建输出文件名，保持原始的段号标识
                output_file = processed_folder / \
                    f"{file_stem}{output_suffix}.sac"

                for retry in range(self.config.max_retries):
                    try:
                        # 读取数据
                        st = read(str(input_file))

                        # 处理每个分量
                        processed_traces = []
                        for tr in st:
                            processed_tr = self._process_single_trace(
                                tr, operations)
                            if processed_tr:
                                processed_traces.append(processed_tr)

                        if processed_traces:
                            # 创建新的数据流并保存
                            processed_st = Stream(traces=processed_traces)
                            processed_st.write(str(output_file), format="SAC")
                            self.logger.info(
                                f"成功处理文件: {input_file.name} -> {output_file.name}")
                            break

                    except Exception as e:
                        self.logger.error(
                            f"处理文件 {input_file.name} 失败: {e}, "
                            f"重试 ({retry + 1}/{self.config.max_retries})"
                        )
                        if retry < self.config.max_retries - 1:
                            delay = self.config.retry_delay_base ** retry
                            time.sleep(delay)
                            continue

                        self.logger.error(f"达到最大重试次数，放弃处理文件 {input_file.name}")

        # 根据配置选择单线程或多线程处理
        if self.config.single_thread:
            for day in days:
                process_single_day(day)
        else:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_day = {
                    executor.submit(process_single_day, day): day
                    for day in days
                }

                for future in as_completed(future_to_day):
                    day = future_to_day[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(
                            f"处理 {day.strftime('%Y-%m-%d')} 时发生错误: {e}")

    def process_pipeline(self,
                         starttime: Union[str, UTCDateTime],
                         endtime: Union[str, UTCDateTime],
                         remove_response: bool = True,
                         merge_segments: bool = False,
                         merge_method: str = "fill_value",
                         fill_value: float = 0,
                         response_output: str = None) -> None:
        """
        完整的数据处理流程

        参数:
            starttime: 起始时间
            endtime: 结束时间
            remove_response: 是否去除仪器响应
            merge_segments: 是否合并数据段
            merge_method: 合并方法，"fill_value" 或 "interpolate"
            fill_value: 填充值，用于 fill_value 方法
        """
        # 转换时间格式
        if isinstance(starttime, str):
            starttime = UTCDateTime(starttime)
        if isinstance(endtime, str):
            endtime = UTCDateTime(endtime)

        days = [
            starttime + datetime.timedelta(days=i)
            for i in range(int((endtime - starttime) // 86400 + 1))
        ]

        # 检查是否已经合并了原始数据
        raw_merged_exists = False
        if self.config.merge_after_download:
            # 检查第一天的merged_raw文件夹是否存在并有合并文件
            day = days[0]
            day_str = day.strftime("%Y-%m-%d")
            merged_raw_folder = self.save_path / day_str / "merged_raw"

            if merged_raw_folder.exists():
                merged_files = list(merged_raw_folder.glob("*.merged.sac"))
                if merged_files:
                    self.logger.info("检测到原始数据已合并，将使用合并后的数据")
                    raw_merged_exists = True

        # 1. 去除仪器响应（如果需要）
        if remove_response:
            # 需要去除仪器响应
            output_type = response_output if response_output else self.config.response_output
            self.logger.info(f"去除仪器响应，输出类型: {output_type}")

            # 构建后缀，包含输出类型信息
            processed_suffix = f"_rmresp_{output_type.lower()}"

            # 如果有合并数据且配置为使用合并数据，则从合并数据中去除响应
            if raw_merged_exists and self.config.merge_after_download:
                self.logger.info("从合并后的原始数据中去除仪器响应")

                self.remove_response_for_data(
                    starttime=starttime,
                    endtime=endtime,
                    processed_suffix=processed_suffix,
                    output=output_type,
                    use_merged=True
                )
            else:
                # 从原始分段数据中去除响应
                self.logger.info("从原始分段数据中去除仪器响应")
                self.remove_response_for_data(
                    starttime=starttime,
                    endtime=endtime,
                    processed_suffix=processed_suffix,
                    output=output_type,
                    use_merged=False
                )

            input_suffix = processed_suffix
            use_merged = False  # 这里设为False是因为已经处理过了，下一步将使用去除响应后的数据
        else:
            input_suffix = ""
            # 不需要去除仪器响应
            if raw_merged_exists and self.config.merge_after_download:
                # 有合并数据且配置为使用合并数据
                self.logger.info("使用下载后立即合并的数据进行处理（不去除响应）")
                use_merged = True
            else:
                # 使用原始分段数据
                self.logger.info("使用原始分段数据进行处理（不去除响应，不合并）")
                use_merged = False

        # 2. 执行数据预处理
        self.process_data(
            starttime=starttime,
            endtime=endtime,
            input_suffix=input_suffix,
            output_suffix="_processed",
            use_merged=use_merged
        )

        # 合并数据部分
        # 如果原始数据已合并且用户要求合并数据段，则禁用后续合并
        if raw_merged_exists and merge_segments:
            # 设置处理时使用合并数据
            use_merged = True

            # 禁用最后的merge_segments步骤，因为我们已经使用了合并的输入
            merge_segments = False
            self.logger.info("已使用合并后的原始数据，跳过最终合并步骤")

        # 3. 合并数据段（如果需要）
        if merge_segments:
            self.logger.info("开始合并数据段")
            merged_count = 0

            for day in days:
                day_str = day.strftime("%Y-%m-%d")

                # 首先检查处理文件夹是否存在
                processed_folder = self.save_path / day_str / "processed"
                if not processed_folder.exists():
                    self.logger.warning(f"处理文件夹不存在: {str(processed_folder)}")

                    # 检查日期文件夹是否存在
                    day_folder = self.save_path / day_str
                    if not day_folder.exists():
                        self.logger.warning(f"日期文件夹不存在: {str(day_folder)}")
                        continue

                    # 如果处理文件夹不存在，尝试在日期文件夹中查找
                    self.logger.info(f"尝试在日期文件夹中查找数据: {str(day_folder)}")

                    # 构建匹配模式
                    if input_suffix and "_processed" in input_suffix:
                        # 已经包含了处理后缀
                        pattern = f"*{input_suffix}.seg*.sac"
                        alt_pattern = f"*{input_suffix}.sac"
                    elif input_suffix:
                        # 有输入后缀，需要添加处理后缀
                        pattern = f"*{input_suffix}_processed.seg*.sac"
                        alt_pattern = f"*{input_suffix}_processed.sac"
                    else:
                        # 无输入后缀，只有处理后缀
                        pattern = f"*_processed.seg*.sac"
                        alt_pattern = f"*_processed.sac"

                    # 尝试在日期文件夹中合并
                    try:
                        self.logger.info(
                            f"在 {str(day_folder)} 中使用模式 {pattern} 或 {alt_pattern}")

                        # 计数文件
                        main_files = list(day_folder.glob(pattern))
                        alt_files = list(day_folder.glob(alt_pattern))

                        if main_files or alt_files:
                            self.logger.info(
                                f"找到文件: {len(main_files)} (主模式) + {len(alt_files)} (替代模式)")

                            # 使用找到文件的模式
                            use_pattern = pattern if main_files else alt_pattern

                            self.merge_daily_segments(
                                day_folder=day_folder,
                                pattern=use_pattern,
                                merge_folder="merged",
                                fill_value=fill_value,
                                merge_method=merge_method
                            )
                            merged_count += 1
                        else:
                            self.logger.warning(
                                f"在 {str(day_folder)} 中没有找到匹配的文件")
                    except Exception as e:
                        self.logger.error(f"合并 {day_str} 的数据时出错: {e}")

                    continue

                # 构建匹配模式
                if input_suffix and "_processed" in input_suffix:
                    # 已经包含了处理后缀
                    pattern = f"*{input_suffix}.seg*.sac"
                    alt_pattern = f"*{input_suffix}.sac"
                elif input_suffix:
                    # 有输入后缀，需要添加处理后缀
                    pattern = f"*{input_suffix}_processed.seg*.sac"
                    alt_pattern = f"*{input_suffix}_processed.sac"
                else:
                    # 无输入后缀，只有处理后缀
                    pattern = f"*_processed.seg*.sac"
                    alt_pattern = f"*_processed.sac"

                # 尝试在处理文件夹中合并
                try:
                    self.logger.info(
                        f"在 {str(processed_folder)} 中使用模式 {pattern} 或 {alt_pattern}")

                    # 计数文件
                    main_files = list(processed_folder.glob(pattern))
                    alt_files = list(processed_folder.glob(alt_pattern))

                    if main_files or alt_files:
                        self.logger.info(
                            f"找到文件: {len(main_files)} (主模式) + {len(alt_files)} (替代模式)")

                        # 使用找到文件的模式
                        use_pattern = pattern if main_files else alt_pattern

                        self.merge_daily_segments(
                            day_folder=processed_folder,
                            pattern=use_pattern,
                            merge_folder="merged",
                            fill_value=fill_value,
                            merge_method=merge_method
                        )
                        merged_count += 1
                    else:
                        self.logger.warning(
                            f"在 {str(processed_folder)} 中没有找到匹配的文件")
                except Exception as e:
                    self.logger.error(f"合并 {day_str} 的数据时出错: {e}")

            self.logger.info(f"合并数据段完成, 共处理了 {merged_count} 天的数据")


def main():
    """
    主函数

    设置配置并启动下载过程的示例
    """
    # 创建下载和处理配置
    config = DownloadConfig(
        network="IM",
        station="H11??",
        channel="EDH",
        max_workers=5,
        need_station_coords=True,    # 启用台站位置信息获取
        merge_after_download=True,   # 启用下载后立即合并
        merge_method="interpolate",  # 使用插值法合并
        merge_fill_value=0,          # 设置填充值（虽然插值模式不使用）
        response_output='DISP'       # 设置去除仪器响应后的输出类型为位移
    )

    process_config = ProcessingConfig(
        freqmin=2.0,
        freqmax=8.0,
        max_spike_width=10,
        spike_threshold=3.0
    )

    # 初始化下载器
    downloader = SeismicDataDownloader(
        config=config,
        save_path="/mnt/f/Data/hydrophone_data/IM_20190501_20190531"
    )

    # 设置处理配置
    downloader.set_processing_config(process_config)

    # 示例 1：下载数据后执行完整处理流程
    starttime = "2019-05-01T00:00:00"
    endtime = "2019-05-31T23:59:59"

    # 1. 下载数据
    downloader.download_data(starttime, endtime)

    # 2. 执行完整处理流程（包括合并）
    downloader.process_pipeline(
        starttime=starttime,
        endtime=endtime,
        remove_response=True,
        merge_segments=True,         # 启用段合并
        merge_method="fill_value",   # 设置合并方法
        fill_value=0                 # 设置填充值
    )

    # 示例 2：只合并特定日期的数据段
    specific_day = "2019-05-15"
    specific_day_datetime = UTCDateTime(specific_day)
    day_str = specific_day_datetime.strftime("%Y-%m-%d")
    day_folder = Path(
        "/mnt/f/Data/hydrophone_data/IM_20190501_20190531") / day_str / "processed"

    if day_folder.exists():
        downloader.merge_daily_segments(
            day_folder=day_folder,
            pattern="*_processed.sac",
            merge_method="interpolate"  # 使用插值合并方法
        )

    # 示例 3：单独执行某些处理操作后再合并
    operations = {
        'remove_mean': True,
        'remove_trend': True,
        'taper': False,  # 不执行尖灭
        'despike': True,
        'bandpass': True
    }

    downloader.process_data(
        starttime=starttime,
        endtime=endtime,
        operations=operations,
        input_suffix="_rmresp",  # 指定输入文件后缀
        output_suffix="_custom"   # 指定输出文件后缀
    )

    # 合并自定义处理后的数据
    for day in [starttime + datetime.timedelta(days=i) for i in range(int((endtime - starttime) // 86400 + 1))]:
        day_str = day.strftime("%Y-%m-%d")
        day_folder = Path(
            "/mnt/f/Data/hydrophone_data/IM_20190501_20190531") / day_str / "processed"

        if day_folder.exists():
            downloader.merge_daily_segments(
                day_folder=day_folder,
                pattern="*_rmresp_custom.sac"
            )


if __name__ == "__main__":
    main()
