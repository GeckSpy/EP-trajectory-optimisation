import numpy as np
import matplotlib.pyplot as plt



z = np.array([-0.2934750138447271, -0.3086100141517818, -0.33945000903622713, -0.3103300142320222, -0.10086500272154808, -0.2420150143734645, -0.0953650027513504, -0.2741900140827056, -0.0953650027513504, -0.2840450098956353, -0.11736500263214111, -0.28555501403752714, -0.2794200140342582, -0.2466249891440384, -0.25215000967727974, -0.3125550140684936, -0.2805549887998495, -0.3228250139800366, -0.21549500679975608, -0.3121900138503406, -0.2974900051631266, -0.24655000348138856, -0.1991300054869498, -0.36868001328548416, -0.2407450052924105, -0.0953650027513504, -0.22810000638855854, -0.26791998898261227, -0.27319000952411443, -0.22972000632580603, -0.22510000637703342, -0.27024500945117325, -0.32968998851720244, -0.25274500964587787, -0.2479750050479197, -0.26265000517014414, -0.3962600094237132, -0.24665000979439355, -0.18498500587884337, -0.2743249888007995, -0.45005501332343556, -0.2504100066726096, -0.26228500958677614, -0.2355700050975429, -0.3073799997509923, -0.2458250141935423, -0.2457850096761831, -0.5727449872356374, -0.5015149874379858, -0.5947849871299695, -0.31205500932992436, -0.11186500266194344, -0.27195999983814545, -0.30669001388014294, -0.33905500936089084, -0.28528501401888207, -0.3341900138766505, -0.35474498831899837, -0.25234001372155035, -0.20326000043860404, -0.313070000425796, -0.33742498797073495, -0.24798000985902036, -0.22985499627247918, -0.5306249872082844, -0.4471250087080989, -0.27360000967019005, -0.3234200092847459, -0.2787850142121897, -0.3021100095138536, -0.29515000488754595, -0.293305001941917, -0.21756000992900226, -0.5115150051642559, -0.5128549942819518, -0.24088000529445708, -0.6990399862334016, -0.5335249972413294, -0.28969000943470746, -0.7071150030460558, -0.3076500137176481, -0.3000300139246974, -0.45601501330384053, -0.20266999267187202, -0.45475000443548197, -0.22504500512150116, -0.25233000085427193, -0.8117650004132884, -0.6020350047692773, -1.4144300032421597, -0.2361650004531839, -0.0953650027513504, -0.2457850096761831, -0.2457850096761831, -0.24238001410412835, -0.25241998900310136, -0.24400000221066875, -0.0953650027513504, -0.18183500313170953, -0.0953650027513504, -0.0953650027513504, -0.4005750047726906, -0.1934700023412006, -0.456545004642976, -0.38484498808247736, -0.24580499754665652, -0.2352200062960037, -0.24174500970548252, -0.24328999574208865, -0.27075000518379966, -0.21208999658119865, -0.2448000053045689, -0.309170006068598, -0.2260449960231199, -0.2466650050409953, -0.3015500058463658, -0.3290550018864451, -0.27292500073235715, -0.0953650027513504, -0.24893999724008609, -0.35262501350371167, -0.23530000071332324, -0.2561100052771508, -0.2679199889898882, -0.3334199964083382, -0.29147501372062834, -0.17312500646949047, -0.24964000531326747, -0.19119999706890667, -0.19799500140652526, -0.2298199981523794, -0.5250699873140547, -0.23557000975415576, -0.10086500272154808, -0.18643000508018304, -0.10086500272154808, -0.30830000959394965, -0.2468800008718972, -0.0953650027513504, -0.26421500345895765, -0.2681050098472042, -0.2301250053133117, -0.42108999841002515, -0.23352000658633187, -0.10086500272154808, -0.237125009742158, -0.7057900018116925, -0.4057650007525808, -0.21053500027483096, -0.37838499496137956, -0.8467749986448325, -0.1644049970927881, -0.24056001415010542, -0.3465700047163409, -0.2432699970813701, -0.2254100051795831, -0.2497750053080381, -0.5521650066875736, -0.40931498792633647, -0.5123799871726078, -0.2410299954135553, -0.2922950006905012, -0.0953650027513504, -0.15616500107717002, -0.18349000002490357, -0.11736500263214111, -0.35035000503557967, -0.0953650027513504, -0.3347899975051405, -0.18978000254719518, -0.5520149960866547, -0.10086500272154808, -0.4152199883683352, -2.3828349827381317, -0.5015049965077196, -0.10636500269174576, -0.11743000870774267, -0.2006850047473563, -0.218560010442161, -0.3519499953908962, -0.39874000206327764, -0.5133450131252175, -1.1767949918939848, -0.19776500168518396, -0.2504750127045554, -0.1994950015359791, -0.13604500045767054, -0.23045000522688497, -0.13024999720073538, -0.0953650027513504, -0.3239649992683553, -0.27375000659230864, -0.2478050021090894, -0.5919100077589974, -0.4701600031548878, -0.10179500861704582, -0.33951999494456686, -0.2505850053130416, -0.26489999766636174, -0.5999949949255097, -0.20336000177485403, -1.1008599889464676, -0.5397450007731095, -0.8494349980828702, -0.13768000082927756, -0.2715000005555339, -0.35250498840468936, -0.5483750075291027, -1.0179850065760547, -0.24061000496294582, -0.37576999721932225, -0.30828500460484065, -0.6758849878169713, -0.6911100077777519, -0.0953650027513504, -0.7583049969107378, 0.552149994087813, -1.6105399930020212, -0.8069999986182665, -0.7884099942384637, -0.2771699947043089, -1.330330003896961, -1.0421949968440458, -0.2201999916433124, -0.0953650027513504, -0.866624986534589, -0.1581500020911335, -0.07310999929177342, -0.18704999764304375, 0.9267599987069843, -0.23366499981057132, -0.24699500214046566, -0.33697998854040634, 0.07693999486218672, -0.8593149895095848, 0.01794499506650027, -0.0953650027513504, -0.324319998959254, -0.277269998769043, -0.15335500189394224, 0.16730000504321652, -0.7219849966932088, -0.13985500036505982, -0.5109200094666448, -1.0187799991690554, -0.2423849963088287, 0.16881499224109575, -0.09445499637513421, -0.36676499738678103, -0.0996899938036222, -0.27039499158127, 0.1670250048773596, -0.22144498739362461, -1.2000699937343597, -0.1556200066479505, 1.212229996577662, -0.10758000906935195, -0.040150000335415825, 0.8451799926842796, 1.0444450039140065, -0.07752500134665752, 0.5189799896397744, 0.2790949971313239, -0.0953650027513504, 1.620770007619285, -0.24006000207009492, -0.2416100021728198, -0.15279999762424268, 0.120214992442925, 0.09241998962534126, 0.1301049946268904, -0.23703499895054847, -0.20447998906456633, -0.13296499731950462, -0.19997999509359943, -0.11543999805871863, 1.9075849961300264, -0.0848900039709406, 0.3661449931678362, -0.3136849959628307, -0.0953650027513504, -0.06585000232007587, 0.1449050029841601, -0.26129000209766673, -0.10086500272154808, -0.23571999625710305, 0.06039498677273514, 0.04373999276867835, 0.38938999966921983, 0.5701149995584274, 0.3166200048945029, -0.3344249955480336, -0.10086500272154808, -0.2064799960135133, 0.42588999492727453, -0.3998500005036476, 0.36384499280393356, -0.2963600065195351, 0.514324991920148, -0.433479999715928, -0.18677999726787675, -0.26027500657801284, -0.21555000542866765, 0.38389000622555614, 0.08686498418683186, -0.0953650027513504, -0.24584500087803463, -0.23566000208666082, 0.13396999896212947, -0.13173999776336132, 1.9786649995148764, -0.17160999905172503, 0.09048499543860089, 0.6503049988023122, 0.7077300056480453, -0.2545650032334379, -0.10341499342030147, 0.681359991816862, -0.5844850003704778, 0.28981000356725417, -0.18174999306938844, 0.09705499040137511, 2.187249988943222, 0.4210450130776735, 1.4294649904259131, -0.0691099960313295, 0.21313499328971375, -0.26507000211131526, -0.17731000197090907, -0.006955011238460429, -0.25607499755278695, -0.22582499700365588, -0.0953650027513504, -0.12125499923422467, 0.028670009502093308, 0.6821599876129767, 0.7778199950917042, -0.20940000074915588, -0.23329000971716596, -0.037794994888827205, 0.5652200065887882, -0.3032500098561286, -0.10086500272154808, -0.1709300023576361, 2.427914990876161, 0.038965000938333105, 0.2653799963518395, 1.2937899935568566, -0.0953650027513504, -0.0953650027513504, 0.5478999914121232, 0.05416999575390946, 0.9316500091154012, -0.0953650027513504, -0.3065449888308649, 0.30578500739648007, -0.07630499797232915, -0.1970899981461116, 0.2326249978650594, 1.0395549969325657, -0.3130599985670415, 0.9152849924357724, -0.0953650027513504, 1.1332150033631478, 1.9510999891572283, -0.08886500552034704, -0.19866998999350471, 0.24758500050666044, -0.07070499903784366, 0.05980499829456676, 1.0672499999564025, 0.09439500331063755, 0.5728549870109418, 0.44260999466496287, 2.42229500412941, 1.875570001458982, -0.060920007977983914, 0.13441499748296337, -0.23801999878196511, -0.2792049961644807, -0.1029449979250785, 0.09060999540815828, -0.3024500048486516, 0.6160600112634711, -0.0953650027513504, 4.664989988232264, -0.035619999704067595, 0.17816501113702543, -0.10166000221215654, -0.0953650027513504, -0.0953650027513504, 0.2942299970163731, 0.20899999172979733, -0.0953650027513504, -0.2509450052966713, 2.315379980398575, 0.41789000343123917, 0.3214149984632968, 0.2408450028160587, 4.528329981883871, -0.10086500272154808, -0.09325500191334868, -0.2456999934729538, -0.01512999376427615, -0.2583500143009587, -0.10636500269174576, -0.2554000065938453, -0.22002499754307792, 0.4103350025397958, -0.09574000843713293, 1.4249049890349852, -0.0833050023575197, -0.13429000314499717, -0.2274700007183128, 2.9527249985621893, -0.13386500254273415, 1.319694994832389, -0.08435000066674547, -0.10086500272154808, 0.2953899938147515, 0.5907599978236249, 0.3820749982842244, -0.28353498904471053, 0.8863299981239834, 2.3936699929035967, 0.1567600037815282, 0.5345699900426553, 0.03073999108892167, -0.13051999935123604, 1.1298949940028251, 0.5320350029651308, 1.8261449855563114, 0.08632500260864617, -0.0660249965658295, 0.22306499146361602, -0.0814800010775798, -0.23964499722933397, -0.23543499765946763, -0.11186500266194344, -0.08543999947869452, 0.643279982781678, 0.30230500351899536, -0.20864500357129145, 1.8637800059805159, 0.7404599942528876, 3.3664799815087463, 0.1610099985191482, 1.9117050012064283, 0.4830649914511014, 3.2845549998019123, 1.430664990657533, 1.2082749866895028, -0.07415499627677491, 1.7435399887690437, 4.014024986579898, 0.2088099988104659, -0.0974150121328421, -0.24379500546638155, 4.0882450088029145, 0.4415899960149545, 1.2963749858536175, -0.23329000977537362, 1.8296149848465575, -0.06258000228990568, 5.756544991279952, -0.08625499789195601, 0.7399900027012336, 2.271529988582188, 1.2134649848812842, 0.7824300001520896, 2.1791049955863855, -0.11415499994473066, 0.11276999337860616, 1.8118449830508325, -0.2285300034491229, 0.17087000109313522, -0.0953650027513504, 0.03503500261285808, 0.09947499876579968, -0.1392600155522814, -0.24705000660469523, -0.20900000161054777, 0.07414000071730698, -0.06720000109635293, 1.4757249900212628, -0.08800000396149699, 0.9349750015171594, 0.7343749990322976, 0.8032199957960984, -0.1273450022417819, 0.17174500306282425, 1.0998849999377853, -0.0953650027513504, -0.0953650027513504, -0.2005099962116219, -0.0953650027513504, 3.2492499912696076, 2.956014974428399, -0.0953650027513504, 0.10604999861243414, 2.274579993201769, 1.1561300033645239, 2.098644987476291, 0.8371999957234948, -0.0953650027513504, -0.29757999988942174, 0.5195599961007247, -0.0953650027513504, 1.6669499875206384, 0.9598000022524502, 0.17390999664348783, -0.10086500272154808, 1.1561700002494035, 0.5546700000704732, 0.0915299902917468, 0.5170750017205137, -0.30436500083305873, 2.857349978767161, 0.024324982630787417, -0.06579500080260914, -0.11186500266194344, -0.12991000142937992, -0.06302499760204228, -0.08066999944276176, -0.0953650027513504, -0.06844500196893932, -0.2398050053670886, -0.0953650027513504, -0.13291000348544912, 2.468334991703159, -0.07046000104310224, 0.17364000161614968, -0.0675449960763217, 0.8548799857671838, -0.10407499625580385, 2.6403699779402814, 3.7894099700497463, 5.838359979847155, 2.1151999977155356, -0.0856150034305756, 5.29062496345432, -0.0774849966619513, 1.918355001922464, -0.2557949974798248, 1.0209599856389104, -0.11688499953743303, 0.486524997788365, -0.17534500051260693, 0.22895499942387687, 0.4494749920486356, -0.2313550034596119, -0.26331499356456334, -0.0953650027513504, 1.9165249942234368, -0.06911500082060229, 1.4535200025929953, 0.1968800017639296, -0.06034999904659344, -0.24702500985586084, -0.0513299996018759, 2.0336749865382444, -0.0439799964078702, 2.611794992131763, 1.1900799855138757, -0.06994000116537791, -0.11186500266194344, 1.8133499883842887, 1.745194984570844, 1.818164991265803, 0.11584999568003695, 3.7870599982270505, -0.24859000983269652, -0.026479996668058448, -0.058444997695914935, 0.9151400006958283, 1.7394099880038993, 1.2638299987811479, -0.07783999905223027, 0.3014450061455136, 1.749714979887358, 0.09882999595720321, -0.08552499898360111, 3.297264993918361, -0.06053000870451797, -0.0910649954166729, 0.07359999808977591, -0.08406500374985626, 0.27577500111510744, 2.540019984218816, 0.24492499824555125, -0.10358999898016918, 0.4644199951799237, -0.0233099990800838, -0.018239997712953482, -0.11838499826262705, 0.8707799809853896, -0.07934000092791393, 1.372994987774291, -0.03867499677289743, -0.03176999955030624, 1.19708497357351, -0.19135001023096265, -0.044639999759965576, 1.108514998551982, 0.6199649886330008, 0.855554985726485, -0.18360501390998252, 0.6933099939415115, -0.18375000097876182, -0.0953650027513504, 2.265954982933181, 0.13402498625509907, -0.0977400010015117, 6.141324989912391, -0.0953650027513504, -0.1922799978783587, 0.16636999063484836, 0.14168998969398672, 5.803574981648126, 0.6170749798693578, 3.435414981089707, 3.8838299827912124, 3.0301649879475008, -0.14243499883741606, -0.2652500065669301, 0.8031649867625674, 0.3866499974683393, -0.0953650027513504, 0.7100849950656993, -0.027564998803427443, -0.11373999924398959, -0.0782999978036969, -0.22908000353345415, 0.7438549915896147, 1.3723199837622815, 1.08577000186051, 0.22310499848390464, 2.8786949805289623, 0.33373500194284134, -0.23260500199103262, -0.22657500975037692, 0.6240899960539537, -0.09891500317462487, 0.07108499336027307, -0.06752499957656255, 0.3327500029554358, 2.309075006429339, -0.2351649958654889, -0.0953650027513504, 5.550319970156124, -0.255500002145709, 0.32368499949370744, -0.0953650027513504, -0.10086500272154808, -0.2127200142640504, -0.0953650027513504, 0.8818749929705518, -0.025024997092259582, 1.2825550002453383, -0.04934499947557924, -0.052425001755182166, 0.8074099941368331, 0.9754000028406153, -0.08263499933673302, -0.0953650027513504, 5.4263549800016335, -0.07826000094064511, -0.19442000047274632, -0.07186000259389402, -0.07094000093638897, 0.5509499949403107, 0.1386999914975604, 0.5178049803798785, -0.06488499780243728, 3.860719981865259, 2.958179983208538, -0.10086500272154808, 1.3220599918495282, 0.08876000011514407, 1.4521649955495377, -0.24464999730844283, -0.020709996999357827, 0.1565850005208631, 1.040009988239035, 2.4601649866453954, 1.1972499939220143, 0.05407499476132216, 0.7062349903135328, 0.19605000055162236, 0.6077899966549012, 0.12208499955158914, 2.146674988369341, 1.9298299924121238, -0.040889999429055024, 2.9726649722724687, -0.23726499605254503, 0.54912499654165, 1.0194849962426815, -0.09551001058571273, -0.12411500253074337, -0.07870499885029858, 0.2936499921634095, -0.025405002030311152, 2.923839995470189, 1.134535006385704, -0.13835500479035545, 0.017335004085907713, 0.23749499980476685, -0.09714500448899344, 2.260139994577912, 0.38632498379593017, -0.03769500222551869, 2.634269981579564, -0.0953650027513504, -0.0739350021322025, 0.20991500633681426, 1.512874983214715, -0.2419750052431482, 0.5528150109166745, -0.20480499704717658, -0.32870999954320723, 0.01599000936403172, 0.8246299972452107, -0.0953650027513504, -0.0953650027513504, 0.1941249957671971, -0.1581899920493015, 0.25990499673207523, 2.889609970137826, 0.7625899946942809, 0.6757399964189972, 0.22574000349413836, 5.584524966674508, 0.16754498267255258, -0.06819000226823846, 0.3249399865453597, -0.15036500245332718, -0.07217500214028405, 0.046249998682469595, -0.24741000655194512, -0.08965501670172671, 3.598739981192921])
u = np.array([i for i in range(len(z))])

plt.figure(figsize=(8, 5))  # Taille du graphique
plt.plot(u, z, color='blue', linewidth=2)

# Ajout de titres et légendes
plt.title("Evolution of the score")
plt.xlabel("Generation")
plt.ylabel("Score")
plt.legend()  # Affiche la légende
plt.grid(True)  # Ajoute une grille
plt.show() 