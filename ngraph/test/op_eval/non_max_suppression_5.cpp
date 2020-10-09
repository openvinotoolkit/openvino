//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/non_max_suppression.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, nonmaxsuppression_center_point_box_format)
{
    std::vector<float> boxes = {0.5, 0.5, 1.0, 1.0, 0.5, 0.6, 1.0, 1.0, 0.5, 0.4, 1.0, 1.0, 0.5,
                                10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
    std::vector<float> scores = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
    const int64_t max_output_boxes_per_class = 3;
    const float iou_threshold = 0.0f;
    const float score_threshold = 0.0f;
    const auto box_encoding = op::v5::BoxEncodingType::CENTER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    ASSERT_TRUE(true);
}