
import logging
import logging.config


def configure_logging():

    logging.config.dictConfig({

        "version": 1,

        "disable_existing_loggers": False,

        # ==================================
        # FORMAT
        # ==================================

        "formatters": {

            "standard": {

                "format": (
                    "%(asctime)s | "
                    "%(levelname)s | "
                    "%(name)s | "
                    "%(message)s"
                ),

                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },


        # ==================================
        # HANDLERS
        # ==================================

        "handlers": {

            "console": {

                "class": "logging.StreamHandler",

                "stream": "ext://sys.stdout",

                "formatter": "standard"
            }
        },


        # ==================================
        # ROOT LOGGER
        # ==================================

        "root": {

            "level": "INFO",

            "handlers": ["console"]
        }
    })
