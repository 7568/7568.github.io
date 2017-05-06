### add propertis to springboot (contain yml and properties file)
---
> this issues has confuse me several days , fortunately now I find a easy way to  resolve it.
> you just create new CustomConfigration class like this:
```

@Configuration
public class CustomConfigration {

    @Autowired
    ConfigurableEnvironment env;

    static YamlPropertiesFactoryBean yaml;

    //load config file use static style
    static {
        yaml = new YamlPropertiesFactoryBean();
        yaml.setResources(new ClassPathResource("config/clog-define.yml"));
    }
    //this annotation can inject properties to spring Environment
    @PostConstruct
    public void setup() throws IOException {
        env.getPropertySources().addLast(new PropertiesPropertySource("custom", yaml.getObject()));
    }

    //this static method will load properties when load java , and inject to spring EnvironmentAware 
    //but this not inject to Envitonment
    @Bean
    public static PropertySourcesPlaceholderConfigurer properties() {
        PropertySourcesPlaceholderConfigurer propertySourcesPlaceholderConfigurer = new PropertySourcesPlaceholderConfigurer();
        propertySourcesPlaceholderConfigurer.setProperties(yaml.getObject());
        return propertySourcesPlaceholderConfigurer;
    }
}
```
> this is my project structures
> ![image help](images/projectstructures.png)
> now the spring boot will load this properties and yml file to enviroment.
> now you can use it like this:
```
    @Value("${operatorLog.managerCode[0]}")
    private Integer addType;
    @Value("${operatorLog.managerCode[1]}")
    private Integer updateType;
    @Value("${operatorLog.managerCode[2]}")
    private Integer deleteType;
```
> or like this :
```
    @Autowired
    Environment env;
    //in method
    env.getProperty("something");
```