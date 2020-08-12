from marshmallow import Schema, fields, validate, ValidationError


# https://github.com/marshmallow-code/marshmallow

def validate_sex(n):
    if n != 'm' and n != 'w' and n != 'n':
        raise ValidationError("Sex should be m,w or n")


def validate_group(n):
    if n != 'EXT' and n != 'LEEXT':
        raise ValidationError("Group should be EXT or LEEXT")


class FDIInputSchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=5))
    assessment_id = fields.Int(required=True, validate=validate.Range(min=1, max=10))

    station = fields.Int(required=True)
    year = fields.Int(required=True)
    month = fields.Int(required=True)
    day = fields.Int(required=True)
    group = fields.Str(required=True, validate=validate_group)
    sex = fields.Str(required=True, validate=validate_sex)
    fish_no = fields.Int(required=True)
    total_length = fields.Int(required=True)
    total_weight = fields.Int(required=True)
    latitude = fields.Float(required=True)
    longitude = fields.Float(required=True)
    bottom_temperature = fields.Float(required=True)
    bottom_salinity = fields.Float(required=True)
    bottom_oxygen_saturation = fields.Float(required=True)
    hydrography_depth = fields.Float(required=True)
    fdi = fields.Float(required=True)


class CFInputSchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=5))
    assessment_id = fields.Int(required=True, validate=validate.Range(min=1, max=10))

    Cryp1 = fields.Int(required=True)
    Cryp2 = fields.Int(required=True)
    Cryp3 = fields.Int(required=True)
    EpPap1 = fields.Int(required=True)
    EpPap2 = fields.Int(required=True)
    EpPap3 = fields.Int(required=True)
    FinRot = fields.Int(required=True)
    Locera1 = fields.Int(required=True)
    Locera2 = fields.Int(required=True)
    Locera3 = fields.Int(required=True)
    PBT = fields.Int(required=True)
    Skel1 = fields.Int(required=True)
    Skel2 = fields.Int(required=True)
    Skel3 = fields.Int(required=True)
    Ulc1 = fields.Int(required=True)
    Ulc2 = fields.Int(required=True)
    Ulc3 = fields.Int(required=True)
    condition_factor = fields.Float(required=True)


class TrainingAPISchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=6))
    action_id = fields.Int(required=True)
    protection_goods_id = fields.Int(required=True)
    user_id = fields.Int(required=True)
    token = fields.Str(required=True)


class MunitionInputSchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=5))
    object_id = fields.Int(required=True)


class LoginInputSchema(Schema):
    email = fields.Str(required=True)
    password = fields.Str(required=True)
