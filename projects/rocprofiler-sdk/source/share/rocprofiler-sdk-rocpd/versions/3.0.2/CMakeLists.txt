# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
#

set(rocpd_schemas data_views.sql rocpd_indexes.sql rocpd_metadata.sql rocpd_tables.sql
                  rocpd_views.sql summary_views.sql)

rocpd_configure_schema_files(VERSION "3.0.2" FILES ${rocpd_schemas})
