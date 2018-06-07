import tensorflow as tf


def build_model_columns(M=10000):
    """Builds a set of wide and deep feature columns."""
    feature_columns = [
        # 2
        tf.feature_column.categorical_column_with_hash_bucket('user.os',5),
        # 309720  100-10
        # tf.feature_column.categorical_column_with_hash_bucket('context.norm_query_hashxitem.mall_city',10 * M),
        # 75495
        # tf.feature_column.categorical_column_with_hash_bucket('context.q2c_infoxitem.mall_distrct',0.5 * M),
        # 10249
        # 9975
        tf.feature_column.categorical_column_with_hash_bucket('item.cat_id_x_user.user_buypower_v2',200000),
        # 26702
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_item.cat_id_x_hour',0.5 * M),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_pv_seg3_x_discrete', 11, default_value=10),
        # 4224
        tf.feature_column.categorical_column_with_hash_bucket('context.relevance_fea_x_bm25',0.1 * M),
        # 353924
        tf.feature_column.categorical_column_with_hash_bucket('item.mall_id_x_user.buy_cnt', 6 * M),
        # 108327
        tf.feature_column.categorical_column_with_hash_bucket('item.mall_id',2 * M),
        # 48330 1-0.1
        # tf.feature_column.categorical_column_with_hash_bucket('context.q2c_infoxitem.mall_city',0.1 * M),
        # 14125
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_item.cat_id_x_day',0.2 * M),
        tf.feature_column.categorical_column_with_identity('context.relevance_fea_x_gTerm3GramRatio_x_discrete_ctr', 11, default_value=10),
        # 257689
        # tf.feature_column.categorical_column_with_hash_bucket('item.mall_idxuser.city',30 * M),
        tf.feature_column.categorical_column_with_identity('user.u_30buyleaf_cat_cnt_x_discrete', 11, default_value=10),
        # 241
        tf.feature_column.categorical_column_with_hash_bucket('user.model',500),
        tf.feature_column.categorical_column_with_identity('item.goods_gender_ctr_seg7_x_discrete_ctr', 11, default_value=10),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_pv_seg1xdiscrete', 11, default_value=10),
        # 6
        # tf.feature_column.categorical_column_with_hash_bucket('context.net_work_type',10),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_click_seg7xdiscrete', 11, default_value=10),
        # 267856
        # tf.feature_column.categorical_column_with_hash_bucket('context.norm_query_hashxcontext.norm_keyword_hash',10 * M),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_click_seg15xdiscrete', 11, default_value=10),
        # 44620
        tf.feature_column.categorical_column_with_hash_bucket('item.cat_id_x_user.province',0.8 * M),
        # 252
        tf.feature_column.categorical_column_with_hash_bucket('user.city',500),
        # 63411
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_item.cat_id_x_day_x_hour',1 * M),
        tf.feature_column.categorical_column_with_identity('context.relevance_fea_x_botCos_x_discrete_ctr', 11, default_value=10),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_pv_seg7xdiscrete', 11, default_value=10),
        # 247620
        tf.feature_column.categorical_column_with_hash_bucket('context.norm_keyword_hash',5 * M),
        tf.feature_column.categorical_column_with_hash_bucket('item.target_gender_group',10),
        # 41975
        # tf.feature_column.categorical_column_with_hash_bucket('context.query_seg_info_list_x_word',200 * M),
        tf.feature_column.categorical_column_with_identity('user.u_30viewleaf_cat_action_cnt_map_x_discrete', 11, default_value=10),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_click_seg60xdiscrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('context.relevance_fea_x_termJaccard_x_discrete_ctr', 11, default_value=10),
        # 5694
        tf.feature_column.categorical_column_with_hash_bucket('item.cat_id_x_user.platform',100000),
        # 285843 5m 1m re
        tf.feature_column.categorical_column_with_hash_bucket('context.norm_query_hash',100 * M),
        # 38669
        # tf.feature_column.categorical_column_with_hash_bucket('item.cat_idxuser.buy_cnt',0.5 * M),
        # 525338
        tf.feature_column.categorical_column_with_hash_bucket('item.ad_id_x_context.norm_query_hash',1000 * M),
        # 275729
        # tf.feature_column.categorical_column_with_hash_bucket('context.q2c_infoxitem.mall_id',2 * M),
        # tf.feature_column.categorical_column_with_identity('item.mall_all_click_seg3xdiscrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_hash_bucket('user.gender',3),
        # 6, 100-50 re
        tf.feature_column.categorical_column_with_hash_bucket('context.match_type',50),
        # 128
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_day_x_hour',2000),
        # 9512 02m 002m
        # tf.feature_column.categorical_column_with_hash_bucket('context.q2c_infoxitem.price_seg',0.02 * M),
        tf.feature_column.categorical_column_with_hash_bucket('item.goods_type',100),
        tf.feature_column.categorical_column_with_identity('item.mall_all_ctr_seg7_x_discrete_ctr', 11, default_value=10),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_pv_seg15xdiscrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('context.relevance_fea_x_twRate_x_discrete_ctr', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('user.hot_buy_cnt_x_discrete', 11, default_value=10),
        # 520766
        # tf.feature_column.categorical_column_with_hash_bucket('item.ad_idxcontext.norm_keyword_hash',2000 * M),
        # 2378
        tf.feature_column.categorical_column_with_hash_bucket('item.cat_id',50000),
        # 152435
        # tf.feature_column.categorical_column_with_hash_bucket('item.cat_idxuser.city',2 * M),
        # tf.feature_column.categorical_column_with_identity('item.mall_all_pv_seg3xdiscrete', 11, default_value=10),
        # 97531 100000 1m
        tf.feature_column.categorical_column_with_hash_bucket('user.u_07viewleaf_cat_days_map_x_item.cat_id',2 * M),
        # 22
        tf.feature_column.categorical_column_with_hash_bucket('item.mall_province',500),
        # 80087
        # tf.feature_column.categorical_column_with_hash_bucket('context.query_seg_info_listxitem.price_segxword',100 * M),
        # tf.feature_column.categorical_column_with_identity('item.mall_all_ctr_seg1xdiscrete_ctr', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('item.goods_platform_ctr_seg7_x_discrete_ctr', 11, default_value=10),
        # 18265 400000 100000 re
        # tf.feature_column.categorical_column_with_hash_bucket('context.q2c_infoxitem.mall_province',100000),
        tf.feature_column.categorical_column_with_identity('user.u_07viewleaf_cat_cnt_x_discrete', 11, default_value=10),
        # 17 10000 1000
        tf.feature_column.categorical_column_with_hash_bucket('item.price',500),
        # 16647
        tf.feature_column.categorical_column_with_hash_bucket('context.title_seg_info_list_x_word',20 * M),
        # tf.feature_column.categorical_column_with_identity('item.mall_all_click_seg1xdiscrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_hash_bucket('context.relevance_fea_x_wordMatch',9),
        # 1444
        # tf.feature_column.categorical_column_with_hash_bucket('item.mall_distrct',3000),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_click_seg90xdiscrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('item.mall_platform_ctr_seg7_x_discrete_ctr', 11, default_value=10),
        # 5562
        tf.feature_column.categorical_column_with_hash_bucket('item.cat_id_x_user.gender',0.1 * M),
        # 390542 10m 1m
        tf.feature_column.categorical_column_with_hash_bucket('item.ad_id_x_user.user_buypower_v2',6 * M),
        tf.feature_column.categorical_column_with_identity('item.ad_all_ctr_seg1_x_discrete_ctr', 11, default_value=10),
        ##  tf.feature_column.categorical_column_with_identity('item.mall_all_pv_seg7xdiscrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('cat.cate_all_ctr_seg1_x_discrete_ctr', 11, default_value=10),
        # 789 check
        # tf.feature_column.categorical_column_with_hash_bucket('context.impr_timexitem.created_at',20 * M),
        # 403309
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_item.ad_id_x_day_x_hour',100 * M),
        # 194534
        tf.feature_column.categorical_column_with_hash_bucket('item.ad_id',2 * M),
        # 534581 2000m 200m
        # tf.feature_column.categorical_column_with_hash_bucket('context.query_seg_info_list_x_context.plan_id_x_word',200 * M),
        # 2509
        # tf.feature_column.categorical_column_with_hash_bucket('user.district',100000),
        # 97 100 10m err
        tf.feature_column.categorical_column_with_hash_bucket('user.u_07viewleaf_cat_days_map_x_item.ad_id',1 * M),
        tf.feature_column.categorical_column_with_identity('context.query_seg_info_list_x_context.title_seg_info_list_x_q_match_ration_x_discrete_ctr', 11, default_value=10),
        # 92 100 10m err
        tf.feature_column.categorical_column_with_hash_bucket('item.mall_id_x_context.norm_query_hash',100 * M),
        # 99 100 1m err
        # tf.feature_column.categorical_column_with_hash_bucket('context.q2c_infoxitem.cat_id',1 * M),
        tf.feature_column.categorical_column_with_identity('user.buy_cnt_x_discrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_hash_bucket('context.relevance_fea_x_xsMatch',4),
        tf.feature_column.categorical_column_with_identity('user.u_30viewleaf_cat_recent_time_map_x_discrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('user.u_07viewgoods_cnt_x_discrete', 11, default_value=10),
        # 100 100 100m err
        # tf.feature_column.categorical_column_with_hash_bucket('item.ad_idxuser.city',100 * M),
        tf.feature_column.categorical_column_with_hash_bucket('context.relevance_fea_x_ppMiss',4),
        tf.feature_column.categorical_column_with_identity('user.u_30viewgoods_cnt_x_discrete', 11, default_value=10),
        # 91 100 20M err
        tf.feature_column.categorical_column_with_hash_bucket('item.ad_id_x_user.buy_cnt', 10 * M),
        # tf.feature_column.categorical_column_with_identity('cat.cate_all_pv_seg3xdiscrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_hash_bucket('item.price_seg_x_user.user_buypower_v2',1000),
        # 76
        tf.feature_column.categorical_column_with_hash_bucket('context.relevance_fea_x_queryLength',10000),
        # 523227
        tf.feature_column.categorical_column_with_hash_bucket('context.query_seg_info_list_x_item.ad_id_x_word',2000 * M),
        # 100 100 1m err
        tf.feature_column.categorical_column_with_hash_bucket('context.plan_id',10 * M),
        # 97 100 3m err
        tf.feature_column.categorical_column_with_hash_bucket('item.ad_id_x_user.platform',3 * M),
        tf.feature_column.categorical_column_with_hash_bucket('item.promotion_type',100),
        # 492396 1000m 100m
        tf.feature_column.categorical_column_with_hash_bucket('context.query_seg_info_list_x_item.mall_id_x_word',1000 * M),
        tf.feature_column.categorical_column_with_identity('cat.cate_platform_ctr_seg7_x_discrete_ctr', 11, default_value=10),
        # 91 100 3m
        tf.feature_column.categorical_column_with_hash_bucket('item.mall_id_x_user.platform',30 * M),
        # 96
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_item.mall_id_x_day',2 * M),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_click_segd_all_click_seg3xdiscrete', 11, default_value=10),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_click_seg30xdiscrete', 11, default_value=10),
        # 98 100 100m
        # tf.feature_column.categorical_column_with_hash_bucket('item.cat_idxcontext.norm_query_hash',100 * M),
        # 4
        tf.feature_column.categorical_column_with_hash_bucket('user.platform',10),
        tf.feature_column.categorical_column_with_identity('cat.cate_province_ctr_seg7_x_discrete_ctr', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('user.u_30buyleaf_cat_recent_time_map_x_discrete', 11, default_value=10),
        # tf.feature_column.categorical_column_with_hash_bucket('user.device',100),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_pv_seg90xdiscrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('user.user_view_searchrate_d7_x_discrete_ctr', 11, default_value=10),
        # 91 100 1m add err
        # tf.feature_column.categorical_column_with_hash_bucket('context.norm_query_hash_x_item.price_seg',10 * M),
        # tf.feature_column.categorical_column_with_identity('cat.cate_all_pv_seg1xdiscrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('context.relevance_fea_x_bowCos_x_discrete_ctr', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('user.u_30buyleaf_cat_action_cnt_map_x_discrete', 11, default_value=10),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_ctr_seg3xdiscrete_ctr', 11, default_value=10),
        # 34
        tf.feature_column.categorical_column_with_hash_bucket('context.relevance_fea_x_titleLength',10000),
        # 529726 100m - 10m re
        # tf.feature_column.categorical_column_with_hash_bucket('context.norm_query_hash_x_context.plan_id',100 * M),
        # tf.feature_column.categorical_column_with_identity('item.mall_all_ctr_seg3xdiscrete_ctr', 11, default_value=10),
        # 271955
        tf.feature_column.categorical_column_with_hash_bucket('item.ad_id_x_user.gender',10 * M),
        tf.feature_column.categorical_column_with_identity('item.ad_all_ctr_seg90_x_discrete_ctr', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('user.u_30favleaf_cat_cnt_x_discrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('user.u_30buygoods_cnt_x_discrete', 11, default_value=10),
        # 4
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_day',200),
        tf.feature_column.categorical_column_with_identity('context.query_seg_info_list_x_context.title_seg_info_list_x_cp_core_match_x_discrete_ctr', 11, default_value=10),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_pv_seg30xdiscrete', 11, default_value=10),
        # 417744
        # tf.feature_column.categorical_column_with_hash_bucket('item.ad_idxuser.os_version',20 * M),
        # tf.feature_column.categorical_column_with_identity('cat.cate_all_pv_seg7xdiscrete', 11, default_value=10),
        # 210120
        tf.feature_column.categorical_column_with_hash_bucket('context.query_seg_info_list_x_context.title_seg_info_list_x_word',2000 * M),
        # tf.feature_column.categorical_column_with_identity('item.mall_all_click_seg7xdiscrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('context.query_seg_info_list_x_context.title_seg_info_list_x_q_match_score_x_discrete_ctr', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('context.relevance_fea_x_qMatchScoreChery_x_discrete_ctr', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('item.mall_province_ctr_seg7_x_discrete_ctr', 11, default_value=10),
        # tf.feature_column.categorical_column_with_hash_bucket('user.provincexitem.mall_province',1000),
        # 7
        # tf.feature_column.categorical_column_with_hash_bucket('context.app_version',10),
        tf.feature_column.categorical_column_with_identity('user.u_30viewleaf_cat_cnt_x_discrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('item.goods_province_ctr_seg7_x_discrete_ctr', 11, default_value=10),
        # tf.feature_column.categorical_column_with_identity('item.mall_all_pv_seg1xdiscrete', 11, default_value=10),
        # 12
        # tf.feature_column.categorical_column_with_hash_bucket('user.manufacture',100),
        tf.feature_column.categorical_column_with_hash_bucket('user.province',900),
        tf.feature_column.categorical_column_with_identity('user.u_07viewleaf_cat_recent_time_map_x_discrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_hash_bucket('user.price_diversity',25),
        tf.feature_column.categorical_column_with_identity('user.user_diff_product_d7_x_discrete', 11, default_value=10),
        # 443637
        tf.feature_column.categorical_column_with_hash_bucket('item.ad_id_x_user.province',8 * M),
        # 142562 200m 2m
        # tf.feature_column.categorical_column_with_hash_bucket('item.salesxuser.buy_cnt',2 * M),
        # 141619
        tf.feature_column.categorical_column_with_hash_bucket('context.query_seg_info_list_x_item.cat_id_x_word',200 * M),
        # 378475 20m 2m
        # tf.feature_column.categorical_column_with_hash_bucket('item.mall_idxuser.os_version',2 * M),
        tf.feature_column.categorical_column_with_hash_bucket('item.event_type',50),
        tf.feature_column.categorical_column_with_hash_bucket('context.relevance_fea_x_cpCoreMatchChery',4),
        tf.feature_column.categorical_column_with_identity('context.relevance_fea_x_wordMatchRatio_x_discrete_ctr', 11, default_value=10),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_click_seg1xdiscrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('cat.cate_all_ctr_seg7_x_discrete_ctr', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('item.sales_x_discrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_hash_bucket('user.user_buypower_v2',25),
        tf.feature_column.categorical_column_with_identity('cat.cate_gender_ctr_seg7_x_discrete_ctr', 11, default_value=10),
        # 20858
        tf.feature_column.categorical_column_with_hash_bucket('item.mall_id_x_user.gender',1 * M),
        tf.feature_column.categorical_column_with_identity('context.relevance_fea_x_qWord3GramRatio_x_discrete_ctr', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('user.u_07buyleaf_cat_cnt_x_discrete', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('item.ad_all_ctr_seg7_x_discrete_ctr', 11, default_value=10),
        # 566
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_item.mall_id_x_day_x_hour',10 * M),
        # 358979
        tf.feature_column.categorical_column_with_hash_bucket('item.mall_id_x_user.province',8 * M ),
        # tf.feature_column.categorical_column_with_identity('cat.cate_all_ctr_seg3xdiscrete_ctr', 11, default_value=10),
        # 32129
        # tf.feature_column.categorical_column_with_hash_bucket('user.cityxitem.mall_city',0.2 * M),
        tf.feature_column.categorical_column_with_identity('context.q2c_info_x_discrete_ctr', 11, default_value=10),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_pv_seg60xdiscrete', 11, default_value=10),
        # 349696 2000m 20m re
        # tf.feature_column.categorical_column_with_hash_bucket('context.norm_query_hashxitem.mall_distrct',200 * M),
        tf.feature_column.categorical_column_with_hash_bucket('item.staple_id',144),
        # 297837
        # tf.feature_column.categorical_column_with_hash_bucket('context.norm_query_hashxitem.mall_province',20 * M),
        tf.feature_column.categorical_column_with_hash_bucket('item.price_seg',25),
        # 2378
        tf.feature_column.categorical_column_with_hash_bucket('cat.cat_id',50000),
        # 14
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_hour',200),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_ctr_seg60xdiscrete_ctr', 11, default_value=10),
        tf.feature_column.categorical_column_with_identity('context.relevance_fea_x_w2vCos_x_discrete_ctr', 11, default_value=10),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_ctr_seg30xdiscrete_ctr', 11, default_value=10),
        # 221
        # tf.feature_column.categorical_column_with_hash_bucket('item.mall_city',500),
        tf.feature_column.categorical_column_with_identity('user.u_07buygoods_cnt_x_discrete', 11, default_value=10),
        # 571
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_item.ad_id_x_hour',36000),
        # 23
        # tf.feature_column.categorical_column_with_hash_bucket('user.os_version',50),
        # tf.feature_column.categorical_column_with_identity('item.ad_all_ctr_seg15xdiscrete_ctr', 11, default_value=10),
        # 9644
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_item.ad_id_x_day',1 * M),
        # 393228
        tf.feature_column.categorical_column_with_hash_bucket('context.impr_time_x_item.mall_id_x_hour',10 * M),
        # 15
        # tf.feature_column.categorical_column_with_hash_bucket('user.genderxitem.target_gender_group',25),
        # 328360 2000m 20m re
        # tf.feature_column.categorical_column_with_hash_bucket('context.norm_query_hash_x_item.cat_id',20 * M),
        tf.feature_column.categorical_column_with_identity('user.u_07viewleaf_cat_action_cnt_map_x_discrete', 11, default_value=10),
        # 292025
        tf.feature_column.categorical_column_with_hash_bucket('item.mall_id_x_user.user_buypower_v2',4 * M),
        # 32780
        # tf.feature_column.categorical_column_with_hash_bucket('item.cat_idxuser.os_version',0.2 * M),
    ]

    return feature_columns, []


def input_fn(data_file, num_epochs=None, shuffle=True, batch_size=1024):
    assert tf.gfile.Exists(data_file), ('%s not found.' % data_file)

    feature_columns, _ = build_model_columns()

    def parse_tfrecord(example):
        label_column = tf.feature_column.numeric_column('label', dtype=tf.float32, default_value=0)
        parsed = tf.parse_single_example(example, features=tf.feature_column.make_parse_example_spec(feature_columns + [label_column]))
        label = parsed.pop('label')
        return parsed, label

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TFRecordDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    dataset = dataset.prefetch(buffer_size=batch_size*10)

    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def input_fn2(data_file, num_epochs=None, shuffle=True, batch_size=1024):
    assert tf.gfile.Exists(data_file), ('%s not found.' % data_file)

    feature_columns, _ = build_model_columns()

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TFRecordDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    dataset = dataset.prefetch(buffer_size=batch_size*10)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset