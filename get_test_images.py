import pathlib
import os
import sqlite3
from PIL import Image
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image as ros_image
from sensor_msgs.msg import PointCloud2
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import open3d


def connect(sqlite_file):
    """Make connection to an SQLite database file."""
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    return conn, c


def close(conn):
    """Close connection to the database."""
    conn.close()


def countRows(cursor, table_name, print_out=False):
    """Returns the total number of rows in the database."""
    cursor.execute("SELECT COUNT(*) FROM {}".format(table_name))
    count = cursor.fetchall()
    if print_out:
        print("\nTotal rows: {}".format(count[0][0]))
    return count[0][0]


def getHeaders(cursor, table_name, print_out=False):
    """Returns a list of tuples with column informations:
    (id, name, type, notnull, default_value, primary_key)
    """
    # Get headers from table "table_name"
    cursor.execute("PRAGMA TABLE_INFO({})".format(table_name))
    info = cursor.fetchall()
    if print_out:
        print("\nColumn Info:\nID, Name, Type, NotNull, DefaultVal, PrimaryKey")
        for col in info:
            print(col)
    return info


def getAllElements(cursor, table_name, print_out=False):
    """Returns a dictionary with all elements of the table database."""
    # Get elements from table "table_name"
    cursor.execute("SELECT * from({}) LIMIT 1000".format(table_name))
    records = cursor.fetchall()
    if print_out:
        print("\nAll elements:")
        for row in records:
            print(row)
    return records


def isTopic(cursor, topic_name, print_out=False):
    """Returns topic_name header if it exists. If it doesn't, returns empty.
    It returns the last topic found with this name.
    """
    boolIsTopic = False
    topicFound = []

    # Get all records for 'topics'
    records = getAllElements(cursor, "topics", print_out=False)

    # Look for specific 'topic_name' in 'records'
    for row in records:
        if row[1] == topic_name:  # 1 is 'name' TODO
            boolIsTopic = True
            topicFound = row
    if print_out:
        if boolIsTopic:
            # 1 is 'name', 0 is 'id' TODO
            print("\nTopic named", topicFound[1], " exists at id ", topicFound[0], "\n")
        else:
            print("\nTopic", topic_name, "could not be found. \n")

    return topicFound


def getAllMessagesInTopic(cursor, topic_name, print_out=False):
    """Returns all timestamps and messages at that topic.
    There is no deserialization for the BLOB data.
    """
    count = 0
    timestamps = []
    messages = []

    # Find if topic exists and its id
    topicFound = isTopic(cursor, topic_name, print_out=False)

    # If not find return empty
    if not topicFound:
        print("Topic", topic_name, "could not be found. \n")
    else:
        records = getAllElements(cursor, "messages", print_out=False)

        # Look for message with the same id from the topic
        for row in records:
            if row[1] == topicFound[0]:  # 1 and 0 is 'topic_id' TODO
                count = count + 1  # count messages for this topic
                timestamps.append(row[2])  # 2 is for timestamp TODO
                messages.append(row[3])  # 3 is for all messages

        # Print
        if print_out:
            print("\nThere are ", count, "messages in ", topicFound[1])

    return timestamps, messages


def save_image(bridge: CvBridge, deserialized, save_path: pathlib.Path):
    cv2_img = np.asarray(bridge.imgmsg_to_cv2(deserialized, "bgra8"))
    cv2.imwrite(save_path.as_posix(), cv2_img)


def save_lidar_points(deserialized, save_path: pathlib.Path) -> None:
    with open(save_path, "wb") as file:
        np.save(file, np.asarray(deserialized.data))


def generate_data(
    db_path: pathlib.Path, save_path: pathlib.Path, topic_name: str, msg_type
):
    if not save_path.is_dir():
        save_path.mkdir(parents=True)
    db, cursor = connect(db_path.as_posix())
    timestamps, messages = getAllMessagesInTopic(cursor, topic_name)
    bridge = CvBridge()
    for timestamp, msg in zip(timestamps, messages):
        deserialized = deserialize_message(msg, msg_type)
        save_lidar_points(
            deserialized, pathlib.Path(save_path.as_posix() + f"/{timestamp}.npy")
        )


if __name__ == "__main__":
    image_path = pathlib.Path("../../test_images")
    lidar_points_path = pathlib.Path("../../lidar_points")
    image_topic = "/zed2i_back/zed_node/left/image_rect_color"
    lidar_topic = "/lidar_points"
    db_path = pathlib.Path("../../data/mental_test/2024_04_09_20_31_57_0.db3")
    generate_data(db_path, lidar_points_path, lidar_topic, PointCloud2)
    # exit()
