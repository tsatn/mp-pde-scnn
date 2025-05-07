# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

###############################################################################
# Reference From PyTorch Core
###############################################################################
from torch.utils.data.datapipes.iter import (
    Batcher,
    Collator,
    Concater,
    Demultiplexer,
    FileLister,
    FileOpener,
    Filter,
    Forker,
    Grouper,
    IterableWrapper,
    Mapper,
    Multiplexer,
    RoutedDecoder,
    Sampler,
    ShardingFilter,
    Shuffler,
    StreamReader,
    UnBatcher,
    Zipper,
)
from torchdata.datapipes.iter.load.aisio import (
    AISFileListerIterDataPipe as AISFileLister,
    AISFileLoaderIterDataPipe as AISFileLoader,
)

###############################################################################
# TorchData
###############################################################################
from torchdata.datapipes.iter.load.fsspec import (
    FSSpecFileListerIterDataPipe as FSSpecFileLister,
    FSSpecFileOpenerIterDataPipe as FSSpecFileOpener,
    FSSpecSaverIterDataPipe as FSSpecSaver,
)

from torchdata.datapipes.iter.load.huggingface import HuggingFaceHubReaderIterDataPipe as HuggingFaceHubReader

from torchdata.datapipes.iter.load.iopath import (
    IoPathFileListerIterDataPipe as IoPathFileLister,
    IoPathFileOpenerIterDataPipe as IoPathFileOpener,
    IoPathSaverIterDataPipe as IoPathSaver,
)

from torchdata.datapipes.iter.load.online import (
    GDriveReaderDataPipe as GDriveReader,
    HTTPReaderIterDataPipe as HttpReader,
    OnlineReaderIterDataPipe as OnlineReader,
)
from torchdata.datapipes.iter.load.s3io import (
    S3FileListerIterDataPipe as S3FileLister,
    S3FileLoaderIterDataPipe as S3FileLoader,
)
from torchdata.datapipes.iter.transform.bucketbatcher import (
    BucketBatcherIterDataPipe as BucketBatcher,
    InBatchShufflerIterDataPipe as InBatchShuffler,
    MaxTokenBucketizerIterDataPipe as MaxTokenBucketizer,
)
from torchdata.datapipes.iter.transform.callable import (
    BatchMapperIterDataPipe as BatchMapper,
    DropperIterDataPipe as Dropper,
    FlatMapperIterDataPipe as FlatMapper,
    FlattenIterDataPipe as Flattener,
    SliceIterDataPipe as Slicer,
)
from torchdata.datapipes.iter.util.bz2fileloader import Bz2FileLoaderIterDataPipe as Bz2FileLoader
from torchdata.datapipes.iter.util.cacheholder import (
    EndOnDiskCacheHolderIterDataPipe as EndOnDiskCacheHolder,
    InMemoryCacheHolderIterDataPipe as InMemoryCacheHolder,
    OnDiskCacheHolderIterDataPipe as OnDiskCacheHolder,
)
from torchdata.datapipes.iter.util.combining import (
    IterKeyZipperIterDataPipe as IterKeyZipper,
    MapKeyZipperIterDataPipe as MapKeyZipper,
    RoundRobinDemultiplexerIterDataPipe as RoundRobinDemultiplexer,
    UnZipperIterDataPipe as UnZipper,
)
from torchdata.datapipes.iter.util.cycler import CyclerIterDataPipe as Cycler, RepeaterIterDataPipe as Repeater
from torchdata.datapipes.iter.util.dataframemaker import (
    DataFrameMakerIterDataPipe as DataFrameMaker,
    ParquetDFLoaderIterDataPipe as ParquetDataFrameLoader,
)
from torchdata.datapipes.iter.util.decompressor import (
    DecompressorIterDataPipe as Decompressor,
    ExtractorIterDataPipe as Extractor,
)
from torchdata.datapipes.iter.util.distributed import FullSyncIterDataPipe as FullSync
from torchdata.datapipes.iter.util.hashchecker import HashCheckerIterDataPipe as HashChecker
from torchdata.datapipes.iter.util.header import HeaderIterDataPipe as Header, LengthSetterIterDataPipe as LengthSetter
from torchdata.datapipes.iter.util.indexadder import (
    EnumeratorIterDataPipe as Enumerator,
    IndexAdderIterDataPipe as IndexAdder,
)
from torchdata.datapipes.iter.util.jsonparser import JsonParserIterDataPipe as JsonParser
from torchdata.datapipes.iter.util.mux_longest import MultiplexerLongestIterDataPipe as MultiplexerLongest
from torchdata.datapipes.iter.util.paragraphaggregator import ParagraphAggregatorIterDataPipe as ParagraphAggregator
from torchdata.datapipes.iter.util.plain_text_reader import (
    CSVDictParserIterDataPipe as CSVDictParser,
    CSVParserIterDataPipe as CSVParser,
    LineReaderIterDataPipe as LineReader,
)
from torchdata.datapipes.iter.util.prefetcher import (
    PinMemoryIterDataPipe as PinMemory,
    PrefetcherIterDataPipe as Prefetcher,
)
from torchdata.datapipes.iter.util.randomsplitter import RandomSplitterIterDataPipe as RandomSplitter
from torchdata.datapipes.iter.util.rararchiveloader import RarArchiveLoaderIterDataPipe as RarArchiveLoader
from torchdata.datapipes.iter.util.rows2columnar import Rows2ColumnarIterDataPipe as Rows2Columnar
from torchdata.datapipes.iter.util.samplemultiplexer import SampleMultiplexerDataPipe as SampleMultiplexer
from torchdata.datapipes.iter.util.saver import SaverIterDataPipe as Saver
from torchdata.datapipes.iter.util.shardexpander import ShardExpanderIterDataPipe as ShardExpander
from torchdata.datapipes.iter.util.sharding import (
    ShardingRoundRobinDispatcherIterDataPipe as ShardingRoundRobinDispatcher,
)
from torchdata.datapipes.iter.util.tararchiveloader import TarArchiveLoaderIterDataPipe as TarArchiveLoader
from torchdata.datapipes.iter.util.tfrecordloader import (
    TFRecordExample,
    TFRecordExampleSpec,
    TFRecordLoaderIterDataPipe as TFRecordLoader,
)
from torchdata.datapipes.iter.util.webdataset import WebDatasetIterDataPipe as WebDataset
from torchdata.datapipes.iter.util.xzfileloader import XzFileLoaderIterDataPipe as XzFileLoader
from torchdata.datapipes.iter.util.zip_longest import ZipperLongestIterDataPipe as ZipperLongest
from torchdata.datapipes.iter.util.ziparchiveloader import ZipArchiveLoaderIterDataPipe as ZipArchiveLoader
from torchdata.datapipes.map.util.converter import MapToIterConverterIterDataPipe as MapToIterConverter

__all__ = [
    "AISFileLister",
    "AISFileLoader",
    "BatchMapper",
    "Batcher",
    "BucketBatcher",
    "Bz2FileLoader",
    "CSVDictParser",
    "CSVParser",
    "Collator",
    "Concater",
    "Cycler",
    "DataFrameMaker",
    "Decompressor",
    "Demultiplexer",
    "Dropper",
    "EndOnDiskCacheHolder",
    "Enumerator",
    "Extractor",
    "FSSpecFileLister",
    "FSSpecFileOpener",
    "FSSpecSaver",
    "FileLister",
    "FileOpener",
    "Filter",
    "FlatMapper",
    "Flattener",
    "Forker",
    "FullSync",
    "GDriveReader",
    "Grouper",
    "HashChecker",
    "Header",
    "HttpReader",
    "HuggingFaceHubReader",
    "InBatchShuffler",
    "InMemoryCacheHolder",
    "IndexAdder",
    "IoPathFileLister",
    "IoPathFileOpener",
    "IoPathSaver",
    "IterDataPipe",
    "IterKeyZipper",
    "IterableWrapper",
    "JsonParser",
    "LengthSetter",
    "LineReader",
    "MapKeyZipper",
    "MapToIterConverter",
    "Mapper",
    "MaxTokenBucketizer",
    "Multiplexer",
    "MultiplexerLongest",
    "OnDiskCacheHolder",
    "OnlineReader",
    "ParagraphAggregator",
    "ParquetDataFrameLoader",
    "PinMemory",
    "Prefetcher",
    "RandomSplitter",
    "RarArchiveLoader",
    "Repeater",
    "RoundRobinDemultiplexer",
    "RoutedDecoder",
    "Rows2Columnar",
    "S3FileLister",
    "S3FileLoader",
    "SampleMultiplexer",
    "Sampler",
    "Saver",
    "ShardExpander",
    "ShardingFilter",
    "ShardingRoundRobinDispatcher",
    "Shuffler",
    "Slicer",
    "StreamReader",
    "TFRecordLoader",
    "TarArchiveLoader",
    "UnBatcher",
    "UnZipper",
    "WebDataset",
    "XzFileLoader",
    "ZipArchiveLoader",
    "Zipper",
    "ZipperLongest",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)

########################################################################################################################
# The part below is generated by parsing through the Python files where IterDataPipes are defined.
# This base template ("__init__.pyi.in") is generated from mypy stubgen with minimal editing for code injection
# The output file will be "__init__.pyi". The generation function is called by "setup.py".
# Note that, for mypy, .pyi file takes precedent over .py file, such that we must define the interface for other
# classes/objects here, even though we are not injecting extra code into them at the moment.

from .util.decompressor import CompressionType
from torchdata._constants import default_timeout_in_s
from torchdata.datapipes.map import MapDataPipe
from torchdata.datapipes.utils import pin_memory_fn
from torch.utils.data import DataChunk, IterableDataset, default_collate
from torch.utils.data.datapipes._typing import _DataPipeMeta
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union, Hashable

try:
    import torcharrow
except ImportError:
    torcharrow = None

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

class IterDataPipe(IterableDataset[T_co], metaclass=_DataPipeMeta):
    functions: Dict[str, Callable] = ...
    reduce_ex_hook: Optional[Callable] = ...
    getstate_hook: Optional[Callable] = ...
    def __getattr__(self, attribute_name: Any): ...
    @classmethod
    def register_function(cls, function_name: Any, function: Any) -> None: ...
    @classmethod
    def register_datapipe_as_function(
        cls, function_name: Any, cls_to_register: Any, enable_df_api_tracing: bool = ...
    ): ...
    def __getstate__(self): ...
    def __reduce_ex__(self, *args: Any, **kwargs: Any): ...
    @classmethod
    def set_getstate_hook(cls, hook_fn: Any) -> None: ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn: Any) -> None: ...
    # Functional form of 'BatcherIterDataPipe'
    def batch(self, batch_size: int, drop_last: bool = False, wrapper_class=DataChunk) -> IterDataPipe: ...
    # Functional form of 'CollatorIterDataPipe'
    def collate(self, conversion: Optional[Union[Callable[..., Any],Dict[Union[str, Any], Union[Callable, Any]],]] = default_collate, collate_fn: Optional[Callable] = None) -> IterDataPipe: ...
    # Functional form of 'ConcaterIterDataPipe'
    def concat(self, *datapipes: IterDataPipe) -> IterDataPipe: ...
    # Functional form of 'DemultiplexerIterDataPipe'
    def demux(self, num_instances: int, classifier_fn: Callable[[T_co], Optional[int]], drop_none: bool = False, buffer_size: int = 1000) -> List[IterDataPipe]: ...
    # Functional form of 'FilterIterDataPipe'
    def filter(self, filter_fn: Callable, input_col=None) -> IterDataPipe: ...
    # Functional form of 'ForkerIterDataPipe'
    def fork(self, num_instances: int, buffer_size: int = 1000) -> List[IterDataPipe]: ...
    # Functional form of 'GrouperIterDataPipe'
    def groupby(self, group_key_fn: Callable[[T_co], Any], *, keep_key: bool = False, buffer_size: int = 10000, group_size: Optional[int] = None, guaranteed_group_size: Optional[int] = None, drop_remaining: bool = False) -> IterDataPipe: ...
    # Functional form of 'FileListerIterDataPipe'
    def list_files(self, masks: Union[str, List[str]] = '', *, recursive: bool = False, abspath: bool = False, non_deterministic: bool = False, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'MapperIterDataPipe'
    def map(self, fn: Callable, input_col=None, output_col=None) -> IterDataPipe: ...
    # Functional form of 'MultiplexerIterDataPipe'
    def mux(self, *datapipes) -> IterDataPipe: ...
    # Functional form of 'FileOpenerIterDataPipe'
    def open_files(self, mode: str = 'r', encoding: Optional[str] = None, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'StreamReaderIterDataPipe'
    def read_from_stream(self, chunk=None) -> IterDataPipe: ...
    # Functional form of 'RoutedDecoderIterDataPipe'
    def routed_decode(self, *handlers: Callable, key_fn: Callable= ...) -> IterDataPipe: ...
    # Functional form of 'ShardingFilterIterDataPipe'
    def sharding_filter(self, sharding_group_filter=None) -> IterDataPipe: ...
    # Functional form of 'ShufflerIterDataPipe'
    def shuffle(self, *, buffer_size: int = 10000, unbatch_level: int = 0) -> IterDataPipe: ...
    # Functional form of 'UnBatcherIterDataPipe'
    def unbatch(self, unbatch_level: int = 1) -> IterDataPipe: ...
    # Functional form of 'ZipperIterDataPipe'
    def zip(self, *datapipes: IterDataPipe) -> IterDataPipe: ...
    # Functional form of 'IndexAdderIterDataPipe'
    def add_index(self, index_name: str = "index") -> IterDataPipe: ...
    # Functional form of 'BucketBatcherIterDataPipe'
    def bucketbatch(self, batch_size: int, drop_last: bool = False, batch_num: int = 100, bucket_num: int = 1, sort_key: Optional[Callable] = None, use_in_batch_shuffle: bool = True) -> IterDataPipe: ...
    # Functional form of 'HashCheckerIterDataPipe'
    def check_hash(self, hash_dict: Dict[str, str], hash_type: str = "sha256", rewind: bool = True) -> IterDataPipe: ...
    # Functional form of 'CyclerIterDataPipe'
    def cycle(self, count: Optional[int] = None) -> IterDataPipe: ...
    # Functional form of 'DataFrameMakerIterDataPipe'
    def dataframe(self, dataframe_size: int = 1000, dtype=None, dtype_generator=None, columns: Optional[List[str]] = None, device: str = "") -> torcharrow.DataFrame: ...
    # Functional form of 'DecompressorIterDataPipe'
    def decompress(self, file_type: Optional[Union[str, CompressionType]] = None) -> IterDataPipe: ...
    # Functional form of 'DropperIterDataPipe'
    def drop(self, indices: Union[Hashable, List[Hashable]]) -> IterDataPipe: ...
    # Functional form of 'EndOnDiskCacheHolderIterDataPipe'
    def end_caching(self, mode="wb", filepath_fn=None, *, same_filepath_fn=False, skip_read=False, timeout=300) -> IterDataPipe: ...
    # Functional form of 'EnumeratorIterDataPipe'
    def enumerate(self, starting_index: int = 0) -> IterDataPipe: ...
    # Functional form of 'FlatMapperIterDataPipe'
    def flatmap(self, fn: Optional[Callable] = None, input_col=None) -> IterDataPipe: ...
    # Functional form of 'FlattenIterDataPipe'
    def flatten(self, indices: Optional[Union[Hashable, List[Hashable]]] = None) -> IterDataPipe: ...
    # Functional form of 'FullSyncIterDataPipe'
    def fullsync(self, timeout=default_timeout_in_s) -> IterDataPipe: ...
    # Functional form of 'HeaderIterDataPipe'
    def header(self, limit: Optional[int] = 10) -> IterDataPipe: ...
    # Functional form of 'InBatchShufflerIterDataPipe'
    def in_batch_shuffle(self) -> IterDataPipe: ...
    # Functional form of 'InMemoryCacheHolderIterDataPipe'
    def in_memory_cache(self, size: Optional[int] = None) -> IterDataPipe: ...
    # Functional form of 'ParagraphAggregatorIterDataPipe'
    def lines_to_paragraphs(self, joiner: Callable= ...) -> IterDataPipe: ...
    # Functional form of 'AISFileListerIterDataPipe'
    def list_files_by_ais(self, url: str, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'FSSpecFileListerIterDataPipe'
    def list_files_by_fsspec(self, masks: Union[str, List[str]] = "", **kwargs) -> IterDataPipe: ...
    # Functional form of 'IoPathFileListerIterDataPipe'
    def list_files_by_iopath(self, masks: Union[str, List[str]] = "", *, pathmgr=None) -> IterDataPipe: ...
    # Functional form of 'S3FileListerIterDataPipe'
    def list_files_by_s3(self, length: int = -1, request_timeout_ms=-1, region="", masks: Union[str, List[str]] = "") -> IterDataPipe: ...
    # Functional form of 'AISFileLoaderIterDataPipe'
    def load_files_by_ais(self, url: str, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'S3FileLoaderIterDataPipe'
    def load_files_by_s3(self, request_timeout_ms=-1, region="", buffer_size=None, multi_part_download=None) -> IterDataPipe: ...
    # Functional form of 'Bz2FileLoaderIterDataPipe'
    def load_from_bz2(self, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'RarArchiveLoaderIterDataPipe'
    def load_from_rar(self, *, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'TarArchiveLoaderIterDataPipe'
    def load_from_tar(self, mode: str = "r:*", length: int = -1) -> IterDataPipe: ...
    # Functional form of 'TFRecordLoaderIterDataPipe'
    def load_from_tfrecord(self, spec: Optional[TFRecordExampleSpec] = None, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'XzFileLoaderIterDataPipe'
    def load_from_xz(self, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'ZipArchiveLoaderIterDataPipe'
    def load_from_zip(self, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'ParquetDFLoaderIterDataPipe'
    def load_parquet_as_df(self, dtype=None, columns: Optional[List[str]] = None, device: str = "", use_threads: bool = False) -> IterDataPipe: ...
    # Functional form of 'BatchMapperIterDataPipe'
    def map_batches(self, fn: Callable, batch_size: int, input_col=None) -> IterDataPipe: ...
    # Functional form of 'MaxTokenBucketizerIterDataPipe'
    def max_token_bucketize(self, max_token_count: int, len_fn: Callable= ..., min_len: int = 0, max_len: Optional[int] = None, buffer_size: int = 1000, include_padding: bool = False) -> IterDataPipe: ...
    # Functional form of '_MemoryCellIterDataPipe'
    def memory_cell(self, remember_elements=1000) -> IterDataPipe: ...
    # Functional form of 'MultiplexerLongestIterDataPipe'
    def mux_longest(self, *datapipes) -> IterDataPipe: ...
    # Functional form of 'OnDiskCacheHolderIterDataPipe'
    def on_disk_cache(self, filepath_fn: Optional[Callable] = None, hash_dict: Dict[str, str] = None, hash_type: str = "sha256", extra_check_fn: Optional[Callable[[str], bool]] = None) -> IterDataPipe: ...
    # Functional form of 'FSSpecFileOpenerIterDataPipe'
    def open_files_by_fsspec(self, mode: str = "r", *, kwargs_for_open: Optional[Dict] = None, **kwargs) -> IterDataPipe: ...
    # Functional form of 'IoPathFileOpenerIterDataPipe'
    def open_files_by_iopath(self, mode: str = "r", pathmgr=None) -> IterDataPipe: ...
    # Functional form of 'CSVParserIterDataPipe'
    def parse_csv(self, *, skip_lines: int = 0, decode: bool = True, encoding: str = "utf-8", errors: str = "ignore", return_path: bool = False, as_tuple: bool = False, **fmtparams) -> IterDataPipe: ...
    # Functional form of 'CSVDictParserIterDataPipe'
    def parse_csv_as_dict(self, *, skip_lines: int = 0, decode: bool = True, encoding: str = "utf-8", errors: str = "ignore", return_path: bool = False, **fmtparams) -> IterDataPipe: ...
    # Functional form of 'JsonParserIterDataPipe'
    def parse_json_files(self, **kwargs) -> IterDataPipe: ...
    # Functional form of 'PinMemoryIterDataPipe'
    def pin_memory(self, device=None, pin_memory_fn=pin_memory_fn) -> IterDataPipe: ...
    # Functional form of 'PrefetcherIterDataPipe'
    def prefetch(self, buffer_size: int = 10) -> IterDataPipe: ...
    # Functional form of 'RandomSplitterIterDataPipe'
    def random_split(self, weights: Dict[T, Union[int, float]], seed, total_length: Optional[int] = None, target: Optional[T] = None) -> Union[IterDataPipe, List[IterDataPipe]]: ...
    # Functional form of 'GDriveReaderDataPipe'
    def read_from_gdrive(self, *, timeout: Optional[float] = None, skip_on_error: bool = False, **kwargs: Optional[Dict[str, Any]]) -> IterDataPipe: ...
    # Functional form of 'HTTPReaderIterDataPipe'
    def read_from_http(self, timeout: Optional[float] = None, skip_on_error: bool = False, **kwargs: Optional[Dict[str, Any]]) -> IterDataPipe: ...
    # Functional form of 'OnlineReaderIterDataPipe'
    def read_from_remote(self, *, timeout: Optional[float] = None, skip_on_error: bool = False, **kwargs: Optional[Dict[str, Any]]) -> IterDataPipe: ...
    # Functional form of 'LineReaderIterDataPipe'
    def readlines(self, *, skip_lines: int = 0, strip_newline: bool = True, decode: bool = False, encoding="utf-8", errors: str = "ignore", return_path: bool = True) -> IterDataPipe: ...
    # Functional form of 'RepeaterIterDataPipe'
    def repeat(self, times: int) -> IterDataPipe: ...
    # Functional form of 'RoundRobinDemultiplexerIterDataPipe'
    def round_robin_demux(self, num_instances: int, buffer_size: int = 1000) -> List[IterDataPipe]: ...
    # Functional form of 'Rows2ColumnarIterDataPipe'
    def rows2columnar(self, column_names: List[str] = None) -> IterDataPipe: ...
    # Functional form of 'FSSpecSaverIterDataPipe'
    def save_by_fsspec(self, mode: str = "w", filepath_fn: Optional[Callable] = None, *, kwargs_for_open: Optional[Dict] = None, **kwargs) -> IterDataPipe: ...
    # Functional form of 'IoPathSaverIterDataPipe'
    def save_by_iopath(self, mode: str = "w", filepath_fn: Optional[Callable] = None, *, pathmgr=None) -> IterDataPipe: ...
    # Functional form of 'SaverIterDataPipe'
    def save_to_disk(self, mode: str = "w", filepath_fn: Optional[Callable] = None) -> IterDataPipe: ...
    # Functional form of 'LengthSetterIterDataPipe'
    def set_length(self, length: int) -> IterDataPipe: ...
    # Functional form of 'ShardExpanderIterDataPipe'
    def shard_expand(self) -> IterDataPipe: ...
    # Functional form of 'ShardingRoundRobinDispatcherIterDataPipe'
    def sharding_round_robin_dispatch(self, sharding_group_filter: Optional[SHARDING_PRIORITIES] = None) -> IterDataPipe: ...
    # Functional form of 'SliceIterDataPipe'
    def slice(self, index: Union[int, List[Hashable]], stop: Optional[int] = None, step: Optional[int] = None) -> IterDataPipe: ...
    # Functional form of 'IterToMapConverterMapDataPipe'
    def to_map_datapipe(self, key_value_fn: Optional[Callable] = None) -> MapDataPipe: ...
    # Functional form of 'UnZipperIterDataPipe'
    def unzip(self, sequence_length: int, buffer_size: int = 1000, columns_to_skip: Optional[Sequence[int]] = None) -> List[IterDataPipe]: ...
    # Functional form of 'WebDatasetIterDataPipe'
    def webdataset(self) -> IterDataPipe: ...
    # Functional form of 'ZipperLongestIterDataPipe'
    def zip_longest(self, *datapipes: IterDataPipe, fill_value: Any = None) -> IterDataPipe: ...
    # Functional form of 'IterKeyZipperIterDataPipe'
    def zip_with_iter(self, ref_datapipe: IterDataPipe, key_fn: Callable, ref_key_fn: Optional[Callable] = None, keep_key: bool = False, buffer_size: int = 10000, merge_fn: Optional[Callable] = None) -> IterDataPipe: ...
    # Functional form of 'MapKeyZipperIterDataPipe'
    def zip_with_map(self, map_datapipe: MapDataPipe, key_fn: Callable, merge_fn: Optional[Callable] = None, keep_key: bool = False) -> IterDataPipe: ...
