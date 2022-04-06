from speedrun import locate


def apply_preprocessing_to_image(image, ch_name, preprocessing_specs):
    if ch_name in preprocessing_specs:
        all_prep_funcs = preprocessing_specs[ch_name]
        all_prep_funcs = all_prep_funcs if isinstance(all_prep_funcs, list) else [all_prep_funcs]
        for prep_fct_specs in all_prep_funcs:
            assert isinstance(prep_fct_specs, dict)
            prep_kwargs = prep_fct_specs["function_kwargs"]
            preprocessing_function = locate(prep_fct_specs["function_name"], [])
            image = preprocessing_function(image, **prep_kwargs)
    return image
