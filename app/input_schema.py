from marshmallow import Schema, fields, validate, ValidationError


# https://github.com/marshmallow-code/marshmallow

def validate_sex(n):
    if n != 'm' and n != 'w' and n != 'n':
        raise ValidationError("Sex should be m,w or n")


def validate_group(n):
    if n != 'EXT' and n != 'LEEXT':
        raise ValidationError("Group should be EXT or LEEXT")


class TrainingAPISchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=6))
    action_id = fields.Int(required=True)
    protection_goods_id = fields.Int(required=True)
    user_id = fields.Str(required=True)
    token = fields.Str(required=True)


class MunitionInputSchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=5))
    object_id = fields.Int(required=True)


class LoginInputSchema(Schema):
    email = fields.Str(required=True)
    password = fields.Str(required=True)
