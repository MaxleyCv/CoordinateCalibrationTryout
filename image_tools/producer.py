class Producer:

    def __init__(self, point_cloud, focal_distance_px: int = 910):
        self.point_cloud = point_cloud
        self.f = focal_distance_px

    def produce_projection(self, X):
        angle = X[-1]
        # TODO: whatever it takes
        return None
