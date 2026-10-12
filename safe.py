class ValidationError(Exception):
    pass

class Validator:

    @staticmethod
    def require_not_null(value,field_name):
        if not value:
            raise ValidationError(f"DONT LEAVE {field_name} BLANK!")
        return value

    @staticmethod
    def require_no_letters(value,field_name):

        if not (value.isdecimal() or value.isascii()):
            raise ValidationError(f"{field_name} must have digits from 0-9 only! No characters or symbols are permitted")
        return value

    @staticmethod
    def require_no_digits(value,field_name):

        if any(c.isdigit() for c in value):
            raise ValidationError(f"{field_name} does not permit continuum numbers")
        return value